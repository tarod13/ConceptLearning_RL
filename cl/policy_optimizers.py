import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy

from utils import one_hot_embedding



class Optimizer:
    def __init__(self, lr=3e-4):
        pass      
    
    def optimize(self, agent, database):
        raise NotImplementedError()



class First_Level_SAC_PolicyOptimizer(Optimizer):
    def __init__(self, learn_alpha=True, batch_size=32, 
                discount_factor=0.99, clip_value=1.0, 
                a_dim=1, lr=3e-4, entropy_update_rate=0.05
                ):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.clip_value = clip_value
        self.min_entropy = -a_dim
        self.lr = lr
        self.entropy_update_rate = entropy_update_rate
        self.H_mean = None
    
    def optimize(self, agent, database): 
        if database.__len__() < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, dones, next_states = \
            database.sample(self.batch_size)      

        # Alias for actor-critic module
        actor_critic = agent.architecture

        # Calculate q-values and action likelihoods
        q1, next_q1_target, q2, next_q2_target, next_log_pa_s, log_alpha = \
            actor_critic(states, actions, next_states)
        alpha = log_alpha.exp()

        # Estimate entropy of the action distributions
        Ha_s = -next_log_pa_s
        Ha_s_mean = Ha_s.detach().mean()

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = Ha_s_mean.item()
        else:
            self.H_mean = Ha_s_mean.item() * self.entropy_update_rate + self.H_mean * (1.0-self.entropy_update_rate)
        
        # Choose minimum next q-value to avoid overestimation of target
        next_q_target = torch.min(next_q1_target, next_q2_target)

        # Calculate next v-value, exactly, with the next action distribution
        next_v_target = next_q_target - alpha * (next_log_pa_s + self.H_mean)

        # Estimate q-value by sampling Bellman expectation
        q_target = rewards + self.discount_factor * next_v_target * (1.-dones)

        # Calculate losses for both critics as the quadratic TD errors
        q1_loss = (q1 - q_target.detach()).pow(2).mean()
        q2_loss = (q2 - q_target.detach()).pow(2).mean()

        # Optimize critics
        actor_critic.q1.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(actor_critic.q1.parameters(), self.clip_value)
        actor_critic.q1.optimizer.step()

        actor_critic.q2.optimizer.zero_grad()
        q2_loss.backward()
        clip_grad_norm_(actor_critic.q2.parameters(), self.clip_value)
        actor_critic.q2.optimizer.step()

        # Calculate temperature loss
        alpha_loss = log_alpha * (Ha_s_mean - self.min_entropy).mean().detach()

        # Optimize temperature parameter alpha (if it is learnable)
        if self.learn_alpha:       
            actor_critic.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([actor_critic.log_alpha], self.clip_value)
            actor_critic.alpha_optimizer.step()      

        # Calculate q-values and action likelihoods again, 
        # but passing gradients through actor
        q1, q2, log_pa_s, log_alpha = actor_critic.evaluate(states)
        alpha = log_alpha.exp()

        # Choose minimum q-value to avoid overestimation
        q = torch.min(q1, q2)

        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        actor_loss = (alpha * (log_pa_s + self.H_mean) - q).mean()

        # Optimize actor
        actor_critic.actor.optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor_critic.actor.parameters(), self.clip_value)
        actor_critic.actor.optimizer.step()

        # Update targets of actor-critic and temperature param.
        actor_critic.update()
        
        metrics = {'q1_loss': q1_loss.item(),
                    'q2_loss': q2_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'alpha_loss': alpha_loss.item(),
                    'alpha': alpha.item(),
                    'base_entropy': self.H_mean
                    }

        return metrics   



class Second_Level_SAC_PolicyOptimizer(Optimizer):
    def __init__(self, learn_alpha=True, batch_size=32, 
                discount_factor=0.99, clip_value=1.0, 
                n_actions=4, init_epsilon=1.0, min_epsilon=0.4, 
                delta_epsilon=2.5e-7, entropy_factor=0.95,
                weight_q_loss = 0.05, entropy_update_rate=0.05, 
                alpha_v_weight=0.1
                ):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.clip_value = clip_value
        self.entropy_target = np.log(n_actions)
        self.epsilon = init_epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = entropy_factor
        self.weight_q_loss = weight_q_loss
        self.alpha_v_weight = alpha_v_weight
        self.entropy_update_rate = entropy_update_rate
        self.H_mean = None
    
    def optimize(self, agents, database, n_step_td=1): 
        if database.__len__() < self.batch_size:
            return None

        # Sample batch
        inner_states, outer_states, actions, rewards, \
            dones, next_inner_states, next_outer_states = \
            database.sample(self.batch_size)      

        n_agents = len(agents)
        q_target = 0.0
        log_softmax_target = 0.0
        HA_s_mean = 0.0
        for i in range(0, n_agents-1):
            q_target_i, log_softmax_target_i, HA_s_mean_i = \
                self.calculate_targets(agents[i], n_step_td, inner_states, 
                                    outer_states, actions, rewards, dones, 
                                    next_inner_states, next_outer_states)
            q_target += (q_target_i - q_target)/(i+1)
            log_softmax_target += (log_softmax_target_i - log_softmax_target)/(i+1)
            HA_s_mean += (HA_s_mean_i - HA_s_mean)/(i+1)

        agent = copy.deepcopy(agents[-1])

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        # Calculate q-values and action likelihoods
        q, next_q, next_PA_s, next_log_PA_s, log_alpha = \
            actor_critic.evaluate_critic(inner_states, outer_states,
                                next_inner_states, next_outer_states)

        alpha = log_alpha.exp().item()

        # Calculate entropy of the action distributions
        HA_s = -(next_PA_s * next_log_PA_s).sum(1, keepdim=True)
        HA_s_mean_last = HA_s.detach().mean()
        HA_s_mean += (HA_s_mean_last - HA_s_mean)/n_agents

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = HA_s_mean.item()
        else:
            self.H_mean = HA_s_mean.item() * self.entropy_update_rate + self.H_mean * (1.0-self.entropy_update_rate)
        
        # Choose minimum next q-value to avoid overestimation of target
        if not actor_critic._parallel:
            next_q_target = torch.min(next_q[0], next_q[1])
        else:
            next_q_target = next_q.min(1)[0]

        # Calculate next v-value, exactly, with the next action distribution
        next_v_target = (next_PA_s * (next_q_target - alpha * (next_log_PA_s + self.H_mean))).sum(1, keepdim=True)

        # Estimate q-value target by sampling Bellman expectation
        q_target_last = rewards + self.discount_factor**n_step_td * next_v_target * (1.-dones)
        q_target += (q_target_last - q_target)/n_agents

        if not actor_critic._parallel:
            # Select q-values corresponding to the action taken
            q1_A = q[0][np.arange(self.batch_size), actions].view(-1,1)
            q2_A = q[1][np.arange(self.batch_size), actions].view(-1,1)

            # Calculate losses for both critics as the quadratic TD errors
            q1_loss = (q1_A - q_target.detach()).pow(2).mean()
            q2_loss = (q2_A - q_target.detach()).pow(2).mean()
            q_loss = q1_loss + q2_loss
        else:
            # Select q-values corresponding to the action taken
            q_A = q[np.arange(self.batch_size),:,actions].view(q.shape[0],q.shape[1],1)

            # Calculate losses for both critics as the quadratic TD errors
            q_loss = (q_A - q_target.unsqueeze(1).detach()).pow(2).mean()

        # Create critic optimizer and optimize model
        actor_critic.q.optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(actor_critic.q.parameters(), self.clip_value)
        actor_critic.q.optimizer.step()    
        
        # Calculate q-values and action likelihoods after critic SGD
        q, PA_s, log_PA_s = actor_critic.evaluate_actor(
            inner_states, outer_states)

        # Choose mean q-value to avoid overestimation
        if not actor_critic._parallel:            
            q_dist = torch.min(q[0], q[1])
        else:
            q_dist = q.min(1)[0]

        # Calculate normalizing factors for target softmax distributions
        z = torch.logsumexp(q_dist/(alpha+1e-10), 1, keepdim=True)

        # Calculate the target log-softmax distribution
        log_softmax_target_last = q_dist/(alpha+1e-10) - z
        log_softmax_target += (log_softmax_target_last - log_softmax_target)/n_agents
        
        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        difference_ratio = alpha * (log_PA_s - log_softmax_target).detach()
        actor_loss = (PA_s * difference_ratio).sum(1, keepdim=True).mean()

        # Create optimizer and optimize model
        actor_critic.actor.optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor_critic.actor.parameters(), self.clip_value)
        actor_critic.actor.optimizer.step()

        # Calculate loss for temperature parameter alpha 
        scaled_min_entropy = self.entropy_target * self.epsilon
        alpha_error = (HA_s_mean - scaled_min_entropy).mean()
        alpha_loss = log_alpha * alpha_error.detach()

        # Optimize temperature (if it is learnable)
        if self.learn_alpha:
            # Create optimizer and optimize model
            actor_critic.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([actor_critic.log_alpha], self.clip_value)
            actor_critic.alpha_optimizer.step()        

        # Update targets of actor-critic and temperature param.
        actor_critic.update()

        # Anneal epsilon
        self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])   

        agents.append(agent) 

        metrics = {'q_loss': q_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'alpha_loss': alpha_loss.item(),
                    'SAC_epsilon': self.epsilon,
                    'alpha': alpha,
                    'base_entropy': self.H_mean,
                    }

        return metrics

    def calculate_targets(self, agent, n_step_td, inner_states, outer_states, 
        actions, rewards, dones, next_inner_states, next_outer_states):

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        with torch.no_grad():
            # Calculate q-values and action likelihoods
            q, next_q, next_PA_s, next_log_PA_s, log_alpha = \
                actor_critic.evaluate_critic(inner_states, outer_states, 
                    next_inner_states, next_outer_states)
            
            alpha = log_alpha.exp().item()

            # Calculate entropy of the action distributions
            HA_s = -(next_PA_s * next_log_PA_s).sum(1, keepdim=True)
            HA_s_mean = HA_s.detach().mean()
            
            # Choose minimum next q-value to avoid overestimation of target
            if not actor_critic._parallel:
                next_q_target = torch.min(next_q[0], next_q[1])
            else:
                next_q_target = next_q.min(1)[0]

            # Calculate next v-value, exactly, with the next action distribution
            next_v_target = (next_PA_s * (next_q_target - alpha * (next_log_PA_s + self.H_mean))).sum(1, keepdim=True)

            # Estimate q-value target by sampling Bellman expectation
            q_target = rewards + self.discount_factor**n_step_td * next_v_target * (1.-dones)

            # Choose mean q-value to avoid overestimation
            if not actor_critic._parallel:
                q_dist = torch.min(q[0], q[1])
            else:
                q_dist = q.min(1)[0]

            # Calculate normalizing factors for target softmax distributions
            z = torch.logsumexp(q_dist/(alpha+1e-10), 1, keepdim=True)

            # Calculate the target log-softmax distribution
            log_softmax_target = q_dist/(alpha+1e-10) - z
            
            return q_target, log_softmax_target, HA_s_mean



class Third_Level_SAC_PolicyOptimizer(Optimizer):
    def __init__(self, learn_alpha=True, batch_size=32, 
                discount_factor=0.99, clip_value=1.0, 
                n_actions=4, init_epsilon=1.0, min_epsilon=0.4, 
                delta_epsilon=2.5e-7, entropy_factor=0.95,
                weight_q_loss = 0.05, entropy_update_rate=0.05, 
                alpha_v_weight=0.1, marginal_update_rate=0.05,
                prior_weight=0.1, batch_size_c=256, prior_loss_type='MSE',
                clip_ratios=False, distributed_contribution=False,
                MC_entropy=False, MC_update_rate=5e-2,
                forgetting_factor=0.1, policy_divergence_limit=0.1
                ):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.batch_size = batch_size
        self.batch_size_c = batch_size_c
        self.discount_factor = discount_factor
        self.clip_value = clip_value
        self.min_entropy = np.log(n_actions)
        self.epsilon = init_epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = entropy_factor
        self.weight_q_loss = weight_q_loss
        self.alpha_v_weight = alpha_v_weight
        self.entropy_update_rate = entropy_update_rate
        self.marginal_update_rate = marginal_update_rate
        self.prior_weight = prior_weight
        self.prior_loss_type = prior_loss_type
        self.clip_ratios = clip_ratios
        self.distributed_contribution = distributed_contribution
        self.MC_entropy = MC_entropy
        self.MC_update_rate = MC_update_rate
        self.forgetting_factor = forgetting_factor
        self.policy_divergence_limit = policy_divergence_limit
        self.H_mean = None
        self.PS = None
    
    def optimize(self, agents, database, n_step_td=1): 
        if database.__len__() < self.batch_size:
            return None

        # Sample batch
        inner_states, outer_states, actions, rewards, \
            dones, next_inner_states, next_outer_states = \
            database.sample(self.batch_size)      

        n_agents = len(agents)
        q_target = 0.0
        log_softmax_target = 0.0
        HA_s_mean = 0.0
        for i in range(0, n_agents-1):
            q_target_i, log_softmax_target_i, HA_s_mean_i = \
                self.calculate_targets(agents[i], n_step_td, inner_states, 
                                    outer_states, actions, rewards, dones, 
                                    next_inner_states, next_outer_states)
            q_target += (q_target_i - q_target)/(i+1)
            log_softmax_target += (log_softmax_target_i - log_softmax_target)/(i+1)
            HA_s_mean += (HA_s_mean_i - HA_s_mean)/(i+1)

        agent = copy.deepcopy(agents[-1])

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        # Calculate q-values and action likelihoods
        q, next_q, next_PA_s, next_log_PA_s, log_alpha = \
            actor_critic.evaluate_critic(inner_states, outer_states,
                                next_inner_states, next_outer_states)

        alpha = log_alpha.exp().item()

        # Calculate entropy of the action distributions
        HA_s = -(next_PA_s * next_log_PA_s).sum(1, keepdim=True)
        HA_s_mean_last = HA_s.detach().mean()
        HA_s_mean += (HA_s_mean_last - HA_s_mean)/n_agents

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = HA_s_mean.item()
        else:
            self.H_mean = HA_s_mean.item() * self.entropy_update_rate + self.H_mean * (1.0-self.entropy_update_rate)
        
        # Choose minimum next q-value to avoid overestimation of target
        if not actor_critic._parallel:
            next_q_target = torch.min(next_q[0], next_q[1])
        else:
            next_q_target = next_q.min(1)[0]

        # Calculate next v-value, exactly, with the next action distribution
        next_v_target = (next_PA_s * (next_q_target - alpha * (next_log_PA_s + self.H_mean))).sum(1, keepdim=True)

        # Estimate q-value target by sampling Bellman expectation
        q_target_last = rewards + self.discount_factor**n_step_td * next_v_target * (1.-dones)
        q_target += (q_target_last - q_target)/n_agents

        if not actor_critic._parallel:
            # Select q-values corresponding to the action taken
            q1_A = q[0][np.arange(self.batch_size), actions].view(-1,1)
            q2_A = q[1][np.arange(self.batch_size), actions].view(-1,1)

            # Calculate losses for both critics as the quadratic TD errors
            q1_loss = (q1_A - q_target.detach()).pow(2).mean()
            q2_loss = (q2_A - q_target.detach()).pow(2).mean()
            q_loss = q1_loss + q2_loss
        else:
            # Select q-values corresponding to the action taken
            q_A = q[np.arange(self.batch_size),:,actions].view(q.shape[0],q.shape[1])

            # Calculate losses for both critics as the quadratic TD errors
            q_loss = (q_A - q_target.unsqueeze(1).detach()).pow(2).mean()

        # Create critic optimizer and optimize model
        actor_critic.q.optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(actor_critic.q.parameters(), self.clip_value)
        actor_critic.q.optimizer.step()    
        
        # Calculate q-values and action likelihoods after critic SGD
        q, PA_s, log_PA_s = actor_critic.evaluate_actor(
            inner_states, outer_states)

        # Choose mean q-value to avoid overestimation
        if not actor_critic._parallel:            
            q_dist = torch.min(q[0], q[1])
        else:
            q_dist = q.min(1)[0]

        # Calculate normalizing factors for target softmax distributions
        z = torch.logsumexp(q_dist/(alpha+1e-10), 1, keepdim=True)

        # Calculate the target log-softmax distribution
        log_softmax_target_last = q_dist/(alpha+1e-10) - z
        log_softmax_target += (log_softmax_target_last - log_softmax_target)/n_agents
        
        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        difference_ratio = alpha * (log_PA_s - log_softmax_target).detach()
        actor_loss = (PA_s * difference_ratio).sum(1, keepdim=True).mean()

        # Alias for concept module
        concept_net = agent.concept_architecture
        PS_s = (concept_net(inner_states, outer_states)[0]).detach()
        PA_S_target, log_PA_S_target = agent.PA_S()

        new_PS = PS_s.mean(0) + 1e-6
        new_PS = new_PS/new_PS.sum()
        if self.PS is None:
            self.PS = new_PS.detach()
        else:
            self.PS = new_PS.detach() * self.marginal_update_rate + self.PS * (1.0-self.marginal_update_rate)
        
        if self.distributed_contribution:
            PA_S = torch.einsum('ij,ik->jk', PS_s, PA_s) + 1e-6
        else:
            concepts = PS_s.argmax(1).detach().cpu().numpy()
            S_one_hot = one_hot_embedding(concepts, PS_s.shape[1])
            PA_S = torch.einsum('ij,ik->jk', S_one_hot, PA_s) + 1e-6
        PA_S = PA_S/PA_S.sum(1, keepdim=True).detach()
        log_PA_S = torch.log(PA_S)
        if self.prior_loss_type == 'MSE':
            prior_loss = (log_PA_S - log_PA_S_target).pow(2).mean()
        else:
            KL_div = (PA_S * (log_PA_S - log_PA_S_target)).sum(1)
            prior_loss = (KL_div * self.PS).sum()

        actor_loss_with_prior = (actor_loss + self.prior_weight * prior_loss) #/ (1.+self.prior_weight)

        # Create optimizer and optimize model
        actor_critic.actor.optimizer.zero_grad()
        actor_loss_with_prior.backward()
        clip_grad_norm_(actor_critic.actor.parameters(), self.clip_value)

        problems = False
        for param in actor_critic.actor.parameters():
            if param.grad is not None:
                problems = problems or not torch.isfinite(param.grad).all()

        assert not problems, 'Explosion!'

        actor_critic.actor.optimizer.step()

        # Calculate loss for temperature parameter alpha 
        scaled_min_entropy = self.min_entropy * self.epsilon
        alpha_error = (HA_s_mean - scaled_min_entropy).mean()
        alpha_loss = log_alpha * alpha_error.detach()

        # Optimize temperature (if it is learnable)
        if self.learn_alpha:
            # Create optimizer and optimize model
            actor_critic.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([actor_critic.log_alpha], self.clip_value)
            actor_critic.alpha_optimizer.step()        

        # Update targets of actor-critic and temperature param.
        actor_critic.update()

        # Anneal epsilon
        self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])   

        agents.append(agent) 

        metrics = {'q_loss': q_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'alpha_loss': alpha_loss.item(),
                    'prior_loss': prior_loss.item(),
                    'SAC_epsilon': self.epsilon,
                    'alpha': alpha,
                    'base_entropy': self.H_mean,
                    }

        return metrics

    def calculate_targets(self, agent, n_step_td, inner_states, outer_states, 
        actions, rewards, dones, next_inner_states, next_outer_states):

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        with torch.no_grad():
            # Calculate q-values and action likelihoods
            q, next_q, next_PA_s, next_log_PA_s, log_alpha = \
                actor_critic.evaluate_critic(inner_states, outer_states, 
                    next_inner_states, next_outer_states)
            
            alpha = log_alpha.exp().item()

            # Calculate entropy of the action distributions
            HA_s = -(next_PA_s * next_log_PA_s).sum(1, keepdim=True)
            HA_s_mean = HA_s.detach().mean()
            
            # Choose minimum next q-value to avoid overestimation of target
            if not actor_critic._parallel:
                next_q_target = torch.min(next_q[0], next_q[1])
            else:
                next_q_target = next_q.min(1)[0]

            # Calculate next v-value, exactly, with the next action distribution
            next_v_target = (next_PA_s * (next_q_target - alpha * (next_log_PA_s + self.H_mean))).sum(1, keepdim=True)

            # Estimate q-value target by sampling Bellman expectation
            q_target = rewards + self.discount_factor**n_step_td * next_v_target * (1.-dones)

            # Choose mean q-value to avoid overestimation
            if not actor_critic._parallel:
                q_dist = torch.min(q[0], q[1])
            else:
                q_dist = q.min(1)[0]

            # Calculate normalizing factors for target softmax distributions
            z = torch.logsumexp(q_dist/(alpha+1e-10), 1, keepdim=True)

            # Calculate the target log-softmax distribution
            log_softmax_target = q_dist/(alpha+1e-10) - z
            
            return q_target, log_softmax_target, HA_s_mean
    
    def optimize_tabular(self, agent, trajectory_buffer, update_target=False): 
        with torch.no_grad():
            N = len(trajectory_buffer.buffer)
            inner_states, outer_states, actions, action_distributions, rewards, dones, next_inner_states, \
                next_outer_states = trajectory_buffer.sample(None, random_sample=False)
            
            PS_s = agent.concept_architecture(inner_states, outer_states)[0]
            concepts = PS_s.argmax(1).detach().cpu().numpy()
            
            next_PS_s = agent.concept_architecture(next_inner_states[-1,:].view(1,-1), next_outer_states[-1,:,:,:].unsqueeze(0))[0]
            next_concept = next_PS_s.argmax(1).detach().cpu().numpy()
            next_concepts = np.concatenate([concepts[1:], next_concept])
            
            PA_S, log_PA_S = agent.PA_S()
            HA_gS = -(PA_S * log_PA_S).sum(1)
            HA_S = (self.PS.view(-1) * HA_gS).sum()
            Alpha = agent.log_Alpha.exp().item()
            
            assert torch.isfinite(HA_S).all(), 'Alahuakbar'

            # PA_s = agent.second_level_architecture.actor(inner_states, outer_states)[0]
            ratios = PA_S[concepts, actions] / action_distributions[np.arange(0,N),actions]
            if self.clip_ratios:
                ratios = ratios.clip(0.0,1.0)

            assert torch.isfinite(ratios).all(), 'Alahuakbar 1'

            Q = (1.-self.forgetting_factor) * agent.Q_table.detach().clone()
            C = (1.-self.forgetting_factor) * agent.C_table.detach().clone()

            assert torch.isfinite(Q).all(), 'Alahuakbar 2'
            assert torch.isfinite(C).all(), 'Alahuakbar 3'

            if N > 0:
                G = 0
                WIS_trajectory = 1
                for i in range(N-1, -1, -1):
                    S, A, R, WIS_step, nS = concepts[i], actions[i], rewards[i], ratios[i], next_concepts[i]
                    G = self.discount_factor * G + R
                    if self.MC_entropy:
                        dH = HA_gS[nS] - HA_S
                        G += self.discount_factor * Alpha * dH 
                    C[S,A] = C[S,A] + WIS_trajectory
                    if torch.is_nonzero(C[S,A]):
                        assert torch.isfinite(C[S,A]), 'Infinity and beyond!'
                        Q[S,A] = Q[S,A] + (WIS_trajectory/C[S,A]) * (G - Q[S,A])
                        WIS_trajectory = WIS_trajectory * WIS_step
                        if self.clip_ratios:
                            WIS_trajectory = WIS_trajectory.clip(0.0,10.0)
                    if not torch.is_nonzero(WIS_trajectory):
                        break 

            dQ = (Q - agent.Q_table).pow(2).mean()

            agent.update_Q(Q, C)
            if update_target:
                agent.update_target(self.MC_update_rate)
            
            Pi = agent.Pi_table.detach().clone()
            log_Pi = torch.log(Pi)
            HA_gS = -(Pi * log_Pi).sum(1)
            HA_S = (self.PS.view(-1) * HA_gS).sum()

            assert torch.isfinite(HA_S).all(), 'Alahuakbar'

            # Optimize Alpha
            agent.update_Alpha(HA_S)

            # Optimize policy
            Alpha = agent.log_Alpha.exp().item()
            duals = (1e-3)*torch.ones_like(self.PS.view(-1,1))
            found_policy = False
            iters_left = 8
            while not found_policy and iters_left > 0:
                Q_adjusted = (Q + Alpha * duals * log_Pi) / (1.+duals)
                Pi_new = torch.exp(Q_adjusted / (Alpha+1e-10))
                Pi_new = Pi_new / Pi_new.sum(1, keepdim=True)
                log_Pi_new = torch.log(Pi_new + 1e-10)
                KL_div = (Pi_new * (log_Pi_new - log_Pi)).sum(1, keepdim=True)
                valid_policies = KL_div <= self.policy_divergence_limit
                if torch.all(valid_policies):
                    found_policy = True
                else:
                    iters_left -= 1
                    duals = 10**(1.-valid_policies.float()) * duals
            
            if found_policy:
                agent.update_policy(Pi_new)
       
            metrics = {
                'Q_change': dQ.item(),
                'entropy': HA_S.item(),
                'Alpha': Alpha,
                'found_policy': float(found_policy),
                'max_dual': duals.max().item(),
            }

            return metrics