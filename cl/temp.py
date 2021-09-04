import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


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
        next_v_target = next_q_target - alpha * (next_log_pa_s + 0.0*self.H_mean)

        # Estimate q-value by sampling Bellman expectation
        q_target = rewards + self.discount_factor * next_v_target * (1.-dones)

        # Calculate losses for both critics as the quadratic TD errors
        q1_loss = (q1 - q_target.detach()).pow(2).mean()
        q2_loss = (q2 - q_target.detach()).pow(2).mean()

        # Calculate loss for temperature parameter alpha (if it is learnable)
        if self.learn_alpha:
            # Calculate temperature loss
            alpha_loss = log_alpha * (Ha_s_mean - self.min_entropy).mean().detach()
        else:
            alpha_loss = 0.0

        loss = q1_loss + q2_loss + alpha_loss

        # Create optimizer and optimize model, with exception of the actor
        optimizer = optim.Adam(agent.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(agent.parameters(), self.clip_value)
        optimizer.step()

        # Calculate q-values and action likelihoods again, 
        # but passing gradients through actor
        q1, q2, log_pa_s, log_alpha = actor_critic.evaluate(states)
        alpha = log_alpha.exp()

        # Choose minimum q-value to avoid overestimation
        q = torch.min(q1, q2)

        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        actor_loss = (alpha * (log_pa_s + 0.0*self.H_mean) - q).mean()

        # Create optimizer and optimize model
        optimizer = optim.Adam(actor_critic.actor.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor_critic.actor.parameters(), self.clip_value)
        optimizer.step()

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


class First_Level_SAC_with_baselines_PolicyOptimizer(Optimizer):
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
        q1, q2, next_v, log_alpha = \
            actor_critic(states, actions, next_states)
        alpha = log_alpha.exp()

        # Estimate q-value by sampling Bellman expectation
        q_target = rewards + self.discount_factor * next_v * (1.-dones)

        # Calculate losses for both critics as the quadratic TD errors
        q1_loss = (q1 - q_target.detach()).pow(2).mean()
        q2_loss = (q2 - q_target.detach()).pow(2).mean()


        critic_loss = q1_loss + q2_loss

        # Create optimizer and optimize critics
        optimizer = optim.Adam(agent.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(agent.parameters(), self.clip_value)
        optimizer.step()

        # Calculate q-values and action likelihoods again, 
        # but passing gradients through actor
        v, q1, q2, log_pa_s, log_alpha = actor_critic.evaluate(states)
        alpha = log_alpha.exp()

        # Estimate entropy of the action distributions
        Ha_s = -log_pa_s
        Ha_s_mean = Ha_s.detach().mean()

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = Ha_s_mean.item()
        else:
            self.H_mean = Ha_s_mean.item() * self.entropy_update_rate + self.H_mean * (1.0-self.entropy_update_rate)

        # Choose minimum q-value to avoid overestimation
        q = torch.min(q1, q2)

        # Estimate v value function
        v_target = q - alpha.detach() * (log_pa_s + 0.0*self.H_mean)

        # Calculate loss for v function
        v_loss = (v - v_target.detach()).pow(2).mean() 

        # Create optimizer and optimize v net
        optimizer = optim.Adam(actor_critic.v.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        v_loss.backward()
        clip_grad_norm_(actor_critic.v.parameters(), self.clip_value)
        optimizer.step()

        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        actor_loss = (-v_target).mean()

        # Create optimizer and optimize model
        optimizer = optim.Adam(actor_critic.actor.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor_critic.actor.parameters(), self.clip_value)
        optimizer.step()

        # Calculate loss for temperature parameter alpha (if it is learnable)
        if self.learn_alpha:
            # Calculate temperature loss
            alpha_loss = log_alpha * (Ha_s_mean - self.min_entropy).mean().detach()
        else:
            alpha_loss = 0.0
        
        # Create optimizer and optimize temperature parameter
        optimizer = optim.Adam([log_alpha], lr=self.lr)  
        optimizer.zero_grad()
        alpha_loss.backward()
        clip_grad_norm_([log_alpha], self.clip_value)
        optimizer.step()

        # Update targets of actor-critic and temperature param.
        actor_critic.update()
        
        metrics = {'q1_loss': q1_loss.item(),
                    'q2_loss': q2_loss.item(),
                    'v_loss': v_loss.item(),
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
                weight_q_loss = 0.05, lr_alpha=3e-4,
                lr=3e-5, entropy_update_rate=0.05
                ):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.clip_value = clip_value
        self.min_entropy = np.log(n_actions) * entropy_factor
        self.epsilon = init_epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = min_epsilon
        self.weight_q_loss = weight_q_loss
        self.lr = lr
        self.lr_alpha = lr_alpha
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
        # for i in range(0, n_agents-1):
        #     q_target_i, log_softmax_target_i, HA_s_mean_i = \
        #         self.calculate_targets(agents[i], inner_states, outer_states, 
        #                             actions, rewards, dones, next_inner_states, 
        #                             next_outer_states)
        #     q_target += (q_target_i - q_target)/(i+1)
        #     log_softmax_target += (log_softmax_target_i - log_softmax_target)/(i+1)
        #     HA_s_mean += (HA_s_mean_i - HA_s_mean)/(i+1)

        agent = copy.deepcopy(agents[-1])
        # agent = agents[-1]

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        # Calculate q-values and action likelihoods
        q, _, PA_s, log_PA_s, log_alpha = \
            actor_critic(inner_states, outer_states)

        _, next_q, next_PA_s, log_next_PA_s, _ = \
            actor_critic(next_inner_states, next_outer_states)
        
        alpha = log_alpha.exp().item()

        # Calculate entropy of the action distributions
        PA = PA_s.mean(0, keepdim=True)
        HA_s = -(PA_s * log_PA_s).sum(1, keepdim=True)
        HA_s_mean_last = HA_s.detach().mean()
        HA_s_mean += (HA_s_mean_last - HA_s_mean)/n_agents

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = HA_s_mean.item()
        else:
            self.H_mean = HA_s_mean.item() * self.entropy_update_rate + self.H_mean * (1.0-self.entropy_update_rate)
        
        # Choose minimum next q-value to avoid overestimation of target
        if not actor_critic.actor_critic_net._parallel:
            next_q_target = torch.min(next_q[0], next_q[1])
        else:
            next_q_target = next_q.min(1)[0]

        # Calculate next v-value, exactly, with the next action distribution
        next_v_target = (next_PA_s * (next_q_target - alpha * (log_next_PA_s + self.H_mean))).sum(1, keepdim=True)

        # Estimate q-value target by sampling Bellman expectation
        q_target_last = rewards + self.discount_factor**n_step_td * next_v_target * (1.-dones)
        q_target += (q_target_last - q_target)/n_agents

        if not actor_critic.actor_critic_net._parallel:
            # Select q-values corresponding to the action taken
            q1_A = q[0][np.arange(self.batch_size), actions].view(-1,1)
            q2_A = q[1][np.arange(self.batch_size), actions].view(-1,1)

            # Calculate losses for both critics as the quadratic TD errors
            q1_loss = (q1_A - q_target.detach()).pow(2).mean()
            q2_loss = (q2_A - q_target.detach()).pow(2).mean()
            q_loss = q1_loss + q2_loss

            # Choose mean q-value to avoid overestimation
            q_dist = torch.min(q[0], q[1])

        else:
            # Select q-values corresponding to the action taken
            q_A = q[np.arange(self.batch_size),:,actions].view(q.shape[0],q.shape[1],1)

            # Calculate losses for both critics as the quadratic TD errors
            q_loss = (q_A - q_target.unsqueeze(1).detach()).pow(2).mean()
            
            # Choose mean q-value to avoid overestimation
            q_dist = q.min(1)[0]

        # Calculate normalizing factors for target softmax distributions
        z = torch.logsumexp(q_dist/(alpha+1e-10), 1, keepdim=True)

        # Calculate the target log-softmax distribution
        log_softmax_target_last = q_dist/(alpha+1e-10) - z
        log_softmax_target += (log_softmax_target_last - log_softmax_target)/n_agents
        
        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        difference_ratio = log_PA_s - log_softmax_target.detach()
        actor_loss = (PA_s * difference_ratio).sum(1, keepdim=True).mean()

        # Calculate total loss
        loss = q_loss*self.weight_q_loss + actor_loss

        # Create optimizer and optimize model
        optimizer = optim.Adam(agent.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(agent.parameters(), self.clip_value)
        optimizer.step()

        # Calculate loss for temperature parameter alpha 
        scaled_min_entropy = self.min_entropy * self.epsilon
        alpha_loss = log_alpha * (HA_s_mean - scaled_min_entropy).mean().detach()

        # Optimize temperature (if it is learnable)
        if self.learn_alpha:
            # Create optimizer and optimize model
            optimizer_alpha = optim.Adam([actor_critic.actor_critic_net.log_alpha], lr=self.lr)  
            optimizer_alpha.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([actor_critic.actor_critic_net.log_alpha], self.clip_value)
            optimizer_alpha.step()        

        # Update targets of actor-critic and temperature param.
        actor_critic.update()

        # Anneal epsilon
        self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])    

        metrics = {'loss': loss.item(),
                    'q_loss': q_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'alpha_loss': alpha_loss.item(),
                    'SAC_epsilon': self.epsilon,
                    'alpha': alpha,
                    'base_entropy': self.H_mean
                    }

        return metrics  
    
    def calculate_targets(self, agent, inner_states, outer_states, 
        actions, rewards, dones, next_inner_states, next_outer_states):

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        with torch.no_grad():
            # Calculate q-values and action likelihoods
            q, _, PA_s, log_PA_s, log_alpha = \
                actor_critic(inner_states, outer_states)

            _, next_q, next_PA_s, log_next_PA_s, _ = \
                actor_critic(next_inner_states, next_outer_states)
            
            alpha = log_alpha.exp().item()

            # Calculate entropy of the action distributions
            PA = PA_s.mean(0, keepdim=True)
            HA_s = -(PA_s * log_PA_s).sum(1, keepdim=True)
            HA_s_mean = HA_s.detach().mean()
            
            # Choose minimum next q-value to avoid overestimation of target
            if not actor_critic.actor_critic_net._parallel:
                next_q_target = torch.min(next_q[0], next_q[1])
            else:
                next_q_target = next_q.min(1)[0]

            # Calculate next v-value, exactly, with the next action distribution
            next_v_target = (next_PA_s * (next_q_target - alpha * (log_next_PA_s + self.H_mean))).sum(1, keepdim=True)

            # Estimate q-value target by sampling Bellman expectation
            q_target = rewards + self.discount_factor**n_step_td * next_v_target * (1.-dones)

            # Choose mean q-value to avoid overestimation
            if not actor_critic.actor_critic_net._parallel:
                q_dist = torch.min(q[0], q[1])
            else:
                q_dist = q.min(1)[0]

            # Calculate normalizing factors for target softmax distributions
            z = torch.logsumexp(q_dist/(alpha+1e-10), 1, keepdim=True)

            # Calculate the target log-softmax distribution
            log_softmax_target = q_dist/(alpha+1e-10) - z
            
            targets = (q_target, log_softmax_target, HA_s_mean)
            return targets


class Second_Level_SAC_with_baselines_PolicyOptimizer(Optimizer):
    def __init__(self, learn_alpha=True, batch_size=32, 
                discount_factor=0.99, clip_value=1.0, 
                n_actions=4, init_epsilon=1.0, min_epsilon=0.4, 
                delta_epsilon=2.5e-7, entropy_factor=0.95,
                weight_q_loss = 0.05, weight_alpha_loss=10.0,
                lr=3e-5, entropy_update_rate=0.05
                ):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.clip_value = clip_value
        self.min_entropy = np.log(n_actions) * entropy_factor
        self.epsilon = init_epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = min_epsilon
        self.weight_q_loss = weight_q_loss
        self.weight_alpha_loss = weight_alpha_loss
        self.lr = lr
        self.entropy_update_rate = entropy_update_rate
        self.H_mean = None
    
    def optimize(self, agent, database): 
        if database.__len__() < self.batch_size:
            return None

        # Sample batch
        inner_states, outer_states, actions, rewards, \
            dones, next_inner_states, next_outer_states = \
            database.sample(self.batch_size)      

        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        # Calculate q-values and action likelihoods
        q1, q2, v, next_v, PA_s, log_PA_s, log_alpha = \
            actor_critic(inner_states, outer_states,
                next_inner_states, next_outer_states)

        alpha = log_alpha.exp()

        # Calculate entropy of the action distributions
        PA = PA_s.mean(0, keepdim=True)
        HA_s = -(PA_s * log_PA_s).sum(1, keepdim=True)
        HA_s_mean = HA_s.detach().mean()

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = HA_s_mean.item()
        else:
            self.H_mean = HA_s_mean.item() * self.entropy_update_rate + self.H_mean * (1.0-self.entropy_update_rate)
        
        # Estimate q-value target by sampling Bellman expectation
        q_target = rewards + self.discount_factor * next_v * (1.-dones)

        # Select q-values corresponding to the action taken
        q1_A = q1[np.arange(self.batch_size), actions].view(-1,1)
        q2_A = q2[np.arange(self.batch_size), actions].view(-1,1)

        # Calculate losses for both critics as the quadratic TD errors
        q1_loss = (q1_A - q_target.detach()).pow(2).mean()
        q2_loss = (q2_A - q_target.detach()).pow(2).mean()

        # Choose minimum q-value to avoid overestimation
        q = torch.min(q1, q2).detach()

        # Calculate normalizing factors for target softmax distributions
        z = torch.logsumexp(q/(alpha+1e-10), 1, keepdim=True)

        # Calculate the target log-softmax distribution
        log_softmax_target = q/(alpha+1e-10) - z
        
        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        difference_ratio = log_PA_s - log_softmax_target.detach()
        actor_loss = (PA_s * difference_ratio).sum(1, keepdim=True).mean()

        # Calculate v-value, exactly, with the action distribution
        v_target = (PA_s * (q - alpha * (log_PA_s + self.H_mean))).sum(1, keepdim=True)

        # Calculate v function loss
        v_loss = (v - v_target.detach()).pow(2).mean()

        # Calculate loss for temperature parameter alpha (if it is learnable)
        if self.learn_alpha:
            # Calculate target entropy
            scaled_min_entropy = self.min_entropy * self.epsilon
            
            # Calculate temperature loss
            alpha_loss = log_alpha * (HA_s_mean - scaled_min_entropy).mean().detach()
        else:
            alpha_loss = 0.0

        # Calculate total loss
        loss = (q1_loss + q2_loss)*self.weight_q_loss + v_loss + actor_loss + alpha_loss*self.weight_alpha_loss

        # Create optimizer and optimize model
        optimizer = optim.Adam(agent.parameters(), lr=self.lr)  
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(agent.parameters(), self.clip_value)
        optimizer.step()

        # Update targets of actor-critic and temperature param.
        actor_critic.update()

        # Anneal epsilon
        self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])    

        metrics = {'loss': loss.item(),
                    'q1_loss': q1_loss.item(),
                    'q2_loss': q2_loss.item(),
                    'v_loss': v_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'alpha_loss': alpha_loss.item(),
                    'SAC_epsilon': self.epsilon,
                    'alpha': alpha.item(),
                    'base_entropy': self.H_mean
                    }

        return metrics   
