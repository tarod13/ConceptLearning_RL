import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.nn.parameter import Parameter
from torch.optim import Adam

from custom_layers import parallel_Linear, Linear_noisy
from policy_nets import *
from q_nets import *
from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")



class v_Net(nn.Module):
    def __init__(self, input_dim, noisy=False):
        super().__init__()        
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)  

        self.l3.apply(weights_init_uniform)
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)


class simple_v_Net(nn.Module):
    def __init__(self, input_dim, noisy=False):
        super().__init__()        
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 1)  

        self.l2.apply(weights_init_uniform)
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = self.l2(x)
        return(x)


class dueling_q_Net(nn.Module):
    def __init__(self, s_dim, n_actions, noisy=False):
        super().__init__() 
        self.s_dim = s_dim
        self.n_actions = n_actions

        if noisy:
            layer = Linear_noisy
        else:
            layer = nn.Linear       

        self.l1 = layer(s_dim, 256)
        self.l2 = layer(256, 256)
        self.lV = layer(256, 1)
        self.lA = layer(256, n_actions)
        
        if noisy:
            self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV.weight, 0.01)
        self.lV.bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA.weight, 0.01)
        self.lA.bias.data.zero_()
        
    def forward(self, s):        
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        V = self.lV(x)        
        A = self.lA(x)
        Q = V.view(-1,1) + A - A.mean(1, keepdim=True) 
        return Q



class actor_critic_with_baselines_Net(nn.Module):
    def __init__(self, s_dim, a_dim, noisy):
        super().__init__()   

        self.s_dim = s_dim
        self.a_dim = a_dim     

        if noisy:
            self.q1 = noisy_q_Net(s_dim, a_dim)        
            self.q2 = noisy_q_Net(s_dim, a_dim)        
            
            self.actor = noisy_actor_Net(s_dim, a_dim) 
        else:
            self.q1 = q_Net(s_dim, a_dim)        
            self.q2 = q_Net(s_dim, a_dim)        
            
            self.actor = actor_Net(s_dim, a_dim)

        self.v = v_Net(s_dim, noisy)            
        self.v_target = v_Net(s_dim, noisy)

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, 0.0)
                
        updateNet(self.v_target, self.v, 1.0)
    
    def forward(self, s, a, next_s):
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        next_v = self.v_target(next_s)
        log_alpha = self.log_alpha.view(-1,1).detach()
        return q1, q2, next_v, log_alpha
        
    def evaluate(self, s):
        v = self.v(s)
        a, log_pa_s = self.actor.sample_actions_and_llhoods(s)
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        log_alpha = self.log_alpha.view(-1,1)
        return v, q1, q2, log_pa_s, log_alpha
    
    def sample_action(self, s, explore=True):
        action = self.actor.sample_action(s.view(1,-1))
        return action
    
    def update(self, rate=5e-3):
        updateNet(self.v_target, self.v, rate)


class discrete_actor_critic_with_baselines_Net(nn.Module):
    def __init__(self, s_dim, n_actions, noisy=True):
        super().__init__()   

        self.s_dim = s_dim
        self.n_actions = n_actions     

        if noisy:
            self.q1 = noisy_dueling_q_Net(s_dim, n_actions)        
            self.q2 = noisy_dueling_q_Net(s_dim, n_actions)
        else:
            self.q1 = dueling_q_Net(s_dim, n_actions)        
            self.q2 = dueling_q_Net(s_dim, n_actions)        
            
        self.actor = softmax_policy_Net(s_dim, n_actions, noisy) 
        self.v = simple_v_Net(s_dim, noisy)
        self.v_target = simple_v_Net(s_dim, noisy)

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, 0.0)
                
        updateNet(self.v_target, self.v, 1.0)
    
    def forward(self, s, next_s):
        q1 = self.q1(s)
        q2 = self.q2(s)
        v = self.v(s)
        next_v = self.v_target(next_s)
        pi, log_pi = self.actor(s)
        log_alpha = self.log_alpha.view(-1,1)
        return q1, q2, v, next_v, pi, log_pi, log_alpha
    
    def sample_action(self, s, explore=True, rng=None):
        PA_s = self.actor(s.view(1,-1))[0].squeeze(0).view(-1)
        if rng is None:
            if explore or np.random.rand() > 0.95:
                A = Categorical(probs=PA_s).sample().item()
            else:
                tie_breaking_dist = torch.isclose(PA_s, PA_s.max()).float()
                tie_breaking_dist /= tie_breaking_dist.sum()
                A = Categorical(probs=tie_breaking_dist).sample().item()  
        else:
            if explore or rng.rand() > 0.95:
                A = rng.choice(self.n_actions, p=PA_s.detach().cpu().numpy())
            else:
                A = PA_s.detach().cpu().argmax().item()
        return A
    
    def sample_actions(self, s, explore=True):
        PA_s = self.actor(s)[0]        
        if explore:
            A = Categorical(probs=PA_s).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PA_s, PA_s.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            A = Categorical(probs=tie_breaking_dist).sample().cpu()                  
        return A
    
    def update(self, rate=5e-3):
        updateNet(self.v_target, self.v, rate)


class vision_actor_critic_with_baselines_Net(nn.Module):
    def __init__(self, s_dim, n_actions, latent_dim=256, noisy=True):
        super().__init__()

        self.vision_net = vision_Net(latent_dim=latent_dim, noisy=noisy)
        self.actor_critic_net = discrete_actor_critic_with_baselines_Net(s_dim + latent_dim, n_actions, noisy=noisy)
    
    def forward(self, inner_state, outer_state,
            next_inner_state, next_outer_state):
        observation = self.observe(inner_state, outer_state)
        next_observation = self.observe(next_inner_state, next_outer_state)
        output = self.actor_critic_net(observation, next_observation)
        return output

    def observe(self, inner_state, outer_state):
        vision_features = self.vision_net(outer_state)
        observation = torch.cat([inner_state, vision_features], dim=1)
        return observation

    def sample_action(self, inner_state, outer_state, explore=True, rng=None):
        # Add batch dimension to state
        inner_state = inner_state.view(1,-1)
        outer_state = outer_state.unsqueeze(0)

        observation = self.observe(inner_state, outer_state)
        A = self.actor_critic_net.sample_action(observation, explore=explore, rng=rng)
        return A
    
    def update(self, rate=5e-3):
        self.actor_critic_net.update(rate)






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



class discrete_actor_critic_Net(nn.Module):
    def __init__(self, s_dim, n_actions, n_heads=8, init_log_alpha=0.0, 
                    noisy=True, parallel=True):
        super().__init__()   

        self.s_dim = s_dim
        self.n_actions = n_actions     
        self._parallel = parallel

        if noisy:
            self.q1 = noisy_dueling_q_Net(s_dim, n_actions)        
            self.q1_target = noisy_dueling_q_Net(s_dim, n_actions)
            self.q2 = noisy_dueling_q_Net(s_dim, n_actions)        
            self.q2_target = noisy_dueling_q_Net(s_dim, n_actions)
        else:
            if not parallel:
                self.q1 = dueling_q_Net(s_dim, n_actions)        
                self.q1_target = dueling_q_Net(s_dim, n_actions)
                self.q2 = dueling_q_Net(s_dim, n_actions)        
                self.q2_target = dueling_q_Net(s_dim, n_actions)
            else:
                self.q = multihead_dueling_q_Net(s_dim, n_actions, n_heads)        
                self.q_target = multihead_dueling_q_Net(s_dim, n_actions, n_heads)
        
        self.actor = softmax_policy_Net(s_dim, n_actions, noisy=noisy) 

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, init_log_alpha)
                
        self.update(rate=1.0)
    
    def forward(self, observations):
        s_pi, s_q, s_q_target = observations
        if not self._parallel:
            q1 = self.q1(s_q[0])
            q2 = self.q2(s_q[1])
            q1_target = self.q1_target(s_q_target[0])
            q2_target = self.q2_target(s_q_target[1])
            q = (q1, q2)
            q_target = (q1_target, q2_target)
        else:
            s = torch.stack(s_q, dim=1)
            q = self.q(s)
            s_target = torch.stack(s_q_target, dim=1)
            q_target = self.q_target(s_target)
        pi, log_pi = self.actor(s_pi)
        log_alpha = self.log_alpha.view(-1,1)
        return q, q_target, pi, log_pi, log_alpha
    
    def evaluate_actor(self, observations):
        s_pi, s_q, s_q_target = observations
        if not self._parallel:
            q1 = self.q1(s_q[0])
            q2 = self.q2(s_q[1])
            q = (q1, q2)            
        else:
            s = torch.stack(s_q, dim=1)
            q = self.q(s)            
        pi, log_pi = self.actor(s_pi)
        log_alpha = self.log_alpha.view(-1,1)
        return q, pi, log_pi
    
    def sample_action(self, s, explore=True, rng=None):
        PA_s = self.actor(s.view(1,-1))[0].squeeze(0).view(-1)
        if rng is None:
            if explore or np.random.rand() > 0.95:
                A = Categorical(probs=PA_s).sample().item()
            else:
                tie_breaking_dist = torch.isclose(PA_s, PA_s.max()).float()
                tie_breaking_dist /= tie_breaking_dist.sum()
                A = Categorical(probs=tie_breaking_dist).sample().item()  
        else:
            if explore or rng.rand() > 0.95:
                A = rng.choice(self.n_actions, p=PA_s.detach().cpu().numpy())
            else:
                A = PA_s.detach().cpu().argmax().item()
        return A, PA_s.detach().cpu().numpy()
    
    def sample_actions(self, s, explore=True):
        PA_s = self.actor(s)[0]        
        if explore:
            A = Categorical(probs=PA_s).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PA_s, PA_s.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            A = Categorical(probs=tie_breaking_dist).sample().cpu()                  
        return A
    
    def update(self, rate=5e-3):
        if not self._parallel:
            updateNet(self.q1_target, self.q1, rate)
            updateNet(self.q2_target, self.q2, rate)
        else:
            updateNet(self.q_target, self.q, rate)


class vision_actor_critic_Net(nn.Module):
    def __init__(self, s_dim, n_actions, n_heads, init_log_alpha=0.0, 
                latent_dim=256, noisy=True, parallel=True):
        super().__init__()

        self.vision_actor_net = vision_Net(latent_dim=latent_dim, noisy=noisy)
        self.vision_critic_nets = nn.ModuleList()
        self.target_vision_critic_nets = nn.ModuleList()
        for head in range(0,n_heads):
            self.vision_critic_nets.append(vision_Net(latent_dim=latent_dim, noisy=noisy))
            self.target_vision_critic_nets.append(vision_Net(latent_dim=latent_dim, noisy=noisy))
        self._n_heads = n_heads
        
        self.actor_critic_net = discrete_actor_critic_Net(s_dim + latent_dim, n_actions, n_heads, 
                                                            init_log_alpha, noisy=noisy, parallel=parallel)
    
    def forward(self, inner_state, outer_state, who='actor'):
        observation = self.get_observations(inner_state, outer_state)
        if who == 'actor':
            output = self.actor_critic_net.evaluate(observation)
        elif who == 'critic':
            output = self.actor_critic_net(observation)
        else:
            raise RuntimeError('Invalid component')
        return output
    
    def get_observations(self, inner_state, outer_state):
        actor_observation = self.get_actor_observations(inner_state, outer_state)
        critic_observation, target_critic_observation = self.get_critic_observations(inner_state, outer_state)
        return actor_observation, critic_observation, target_critic_observation

    def get_actor_observations(self, inner_state, outer_state):
        vision_features_actor = self.vision_actor_net(outer_state)
        actor_observation = torch.cat([inner_state, vision_features_actor], dim=1)
        return actor_observation

    def get_critic_observations(self, inner_state, outer_state):
        critic_observations = []
        target_critic_observations = []
        for head in range(0,self._n_heads):
            vision_features_critic = self.vision_critic_nets[head](outer_state)
            critic_observations.append(torch.cat([inner_state, vision_features_critic], dim=1))
            target_vision_features_critic = self.target_vision_critic_nets[head](outer_state)
            target_critic_observations.append(torch.cat([inner_state, target_vision_features_critic], dim=1))        
        return critic_observations, target_critic_observations 

    def sample_action(self, inner_state, outer_state, explore=True, rng=None):
        # Add batch dimension to state
        inner_state = inner_state.view(1,-1)
        outer_state = outer_state.unsqueeze(0)

        actor_observation = self.get_actor_observations(inner_state, outer_state)
        A, PA_s = self.actor_critic_net.sample_action(actor_observation, explore=explore, rng=rng)
        return A, PA_s
    
    def update(self, rate=5e-3):
        self.actor_critic_net.update(rate)
        for head in range(0,self._n_heads):
            updateNet(self.target_vision_critic_nets[head], self.vision_critic_nets[head], rate)
    
    def get_alpha(self):
        return self.actor_critic_net.log_alpha.exp().item()







    def optimize_tabular(self, agent, database): 
        if database.__len__() < self.batch_size_c:
            return None

        # Sample batch
        inner_states, outer_states, actions, rewards, \
            dones, next_inner_states, next_outer_states = \
            database.sample(self.batch_size_c)
        
        Q1 = agent.Q_table1
        Q2 = agent.Q_table2
        Q1_target = agent.Q_table1_target
        Q2_target = agent.Q_table2_target

        next_Q = torch.min(Q1_target, Q2_target)
        Alpha = agent.log_Alpha.exp()
        
        PS_s = agent.concept_architecture(inner_states, outer_states)[0]
        next_PS_s = agent.concept_architecture(next_inner_states, next_outer_states)[0]
        S = PS_s.argmax(1).detach().cpu().numpy()
        next_S = next_PS_s.argmax(1).detach().cpu().numpy()

        PA_S, log_PA_S = agent.PA_S()

        # PA_s = torch.einsum('ij,jk->ik', PS_s, PA_S)
        PA_s = PA_S[S,:]
        HA_S = -(PA_s * torch.log(PA_s)).sum(1).mean()

        agent.update_mean_entropy(HA_S)
        H_mean = agent.H_mean

        next_V_S = (PA_S * (next_Q - Alpha * (log_PA_S + H_mean))).sum(1)
        # next_V = torch.einsum('ij,j->i', next_PS_s, next_V_S).view(-1,1)
        next_V = next_V_S[next_S].view(-1,1)
        Q_target = rewards + self.discount_factor * next_V * (1.0-dones)

        mask = torch.randint_like(Q1, high=2).float()
        Q = mask * Q1 + (1-mask) * Q2
        # errors = (Q[:,actions].T - Q_target.detach())**2
        # Q_loss = torch.einsum('ij,ij->i', PS_s, errors).mean()
        # print(S.shape, actions.shape, Q[S,actions].shape)
        Q_loss = ((Q[S,actions].view(-1,1) - Q_target.detach())**2).mean()

        # Optimize Q tables
        agent.Q_optimizer.zero_grad()
        Q_loss.backward()
        clip_grad_norm_([Q1, Q2], self.clip_value)
        agent.Q_optimizer.step()

        # Optimize Alpha
        agent.update_Alpha(HA_S)

        # Update targets
        agent.update_targets()

        metrics = {
            'Q_loss': Q_loss.item(),
            'entropy': HA_S.item(),
            'Alpha': Alpha.item(),
        }

        return metrics

    

        def optimize_tabular(self, agent, trajectory_buffer): 
        with torch.no_grad():
            N = len(trajectory_buffer.buffer)
            inner_states, outer_states, actions, rewards, dones, next_inner_states, \
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

            PA_s = agent.second_level_architecture.actor(inner_states, outer_states)[0]
            ratios = PA_S[concepts, actions] / PA_s[np.arange(0,N),actions]

            assert torch.isfinite(ratios).all(), 'Alahuakbar 1'

            Q = agent.Q_table.detach().clone()
            C = agent.C_table.detach().clone()
            Q0 = Q.clone()

            assert torch.isfinite(Q).all(), 'Alahuakbar 2'
            assert torch.isfinite(C).all(), 'Alahuakbar 3'

            if N > 0:
                G = 0
                WIS_trajectory = 1
                for i in range(N-1, -1, -1):
                    S, A, R, nS = concepts[i], actions[i], rewards[i], next_concepts[i]
                    dH = HA_gS[nS] - HA_S
                    G = self.discount_factor * G + R + self.discount_factor * Alpha * dH 
                    C[S,A] = C[S,A] + WIS_trajectory
                    if torch.is_nonzero(C[S,A]):
                        assert torch.isfinite(C[S,A]), 'Infinity and beyond!'
                        Q[S,A] = Q[S,A] + (WIS_trajectory/C[S,A]) * (G - Q[S,A])
                        agent.update_Q(Q, C)

                        PA_S = agent.PA_S()[0]
                        HA_gS = -(PA_S * log_PA_S).sum(1)
                        HA_S = (self.PS.view(-1) * HA_gS).sum()

                        WIS_step = PA_S[S,A] / PA_s[i,A]
                        WIS_trajectory = WIS_trajectory * WIS_step
                    if not torch.is_nonzero(WIS_trajectory):
                        break 

            agent.update_Q(Q, 0.9*C)
            dQ = (Q - Q0).pow(2).mean()            

            PA_S, log_PA_S = agent.PA_S()
            HA_gS = -(PA_S * log_PA_S).sum(1)
            HA_S = (self.PS.view(-1) * HA_gS).sum()

            assert torch.isfinite(HA_S).all(), 'Alahuakbar'

            # Optimize Alpha
            agent.update_Alpha(HA_S)

       
            metrics = {
                'Q_change': dQ.item(),
                'entropy': HA_S.item(),
                'Alpha': Alpha,
            }

            return metrics