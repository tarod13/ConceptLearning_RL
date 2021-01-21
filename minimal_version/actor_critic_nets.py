import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.nn.parameter import Parameter

from custom_layers import parallel_Linear, Linear_noisy
from policy_nets import softmax_policy_Net, actor_Net, noisy_actor_Net, s_Net
from q_nets import q_Net, dueling_q_Net, noisy_q_Net, noisy_dueling_q_Net, v_Net, simple_v_Net , multihead_dueling_q_Net
from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class actor_critic_Net(nn.Module):
    def __init__(self, s_dim, a_dim, noisy):
        super().__init__()   

        self.s_dim = s_dim
        self.a_dim = a_dim     

        if noisy:
            self.q1 = noisy_q_Net(s_dim, a_dim)        
            self.q1_target = noisy_q_Net(s_dim, a_dim)
            self.q2 = noisy_q_Net(s_dim, a_dim)        
            self.q2_target = noisy_q_Net(s_dim, a_dim)
        
            self.actor = noisy_actor_Net(s_dim, a_dim) 
        else:
            self.q1 = q_Net(s_dim, a_dim)        
            self.q1_target = q_Net(s_dim, a_dim)
            self.q2 = q_Net(s_dim, a_dim)        
            self.q2_target = q_Net(s_dim, a_dim)
        
            self.actor = actor_Net(s_dim, a_dim)

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, 0.0)
                
        updateNet(self.q1_target, self.q1, 1.0)
        updateNet(self.q2_target, self.q2, 1.0)
    
    def forward(self, s, a, next_s):
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        next_a, next_log_pa_s = self.actor.sample_actions_and_llhoods(next_s)
        next_q1_target = self.q1_target(next_s, next_a.detach())
        next_q2_target = self.q2_target(next_s, next_a.detach())
        log_alpha = self.log_alpha.view(-1,1)
        return q1, next_q1_target, q2, next_q2_target, \
            next_log_pa_s, log_alpha
        
    def evaluate(self, s):
        a, log_pa_s = self.actor.sample_actions_and_llhoods(s)
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        log_alpha = self.log_alpha.view(-1,1).detach()
        return q1, q2, log_pa_s, log_alpha
    
    def sample_action(self, s, explore=True):
        action = self.actor.sample_action(s.view(1,-1))
        return action
    
    def update_targets(self, rate):
        updateNet(self.q1_target, self.q1, rate)
        updateNet(self.q2_target, self.q2, rate)
    
    def update(self, rate=5e-3):
        self.update_targets(rate)


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


class discrete_actor_critic_Net(nn.Module):
    def __init__(self, s_dim, n_actions, n_heads=8, init_log_alpha=0.0, noisy=True):
        super().__init__()   

        self.s_dim = s_dim
        self.n_actions = n_actions     

        if noisy:
            self.q1 = noisy_dueling_q_Net(s_dim, n_actions)        
            self.q1_target = noisy_dueling_q_Net(s_dim, n_actions)
            self.q2 = noisy_dueling_q_Net(s_dim, n_actions)        
            self.q2_target = noisy_dueling_q_Net(s_dim, n_actions)
        else:
            self.q1 = dueling_q_Net(s_dim, n_actions)        
            self.q1_target = dueling_q_Net(s_dim, n_actions)
            self.q2 = dueling_q_Net(s_dim, n_actions)        
            self.q2_target = dueling_q_Net(s_dim, n_actions)
            # self.q = multihead_dueling_q_Net(s_dim, n_actions, n_heads)        
            # self.q_target = multihead_dueling_q_Net(s_dim, n_actions, n_heads)
        
        self.actor = softmax_policy_Net(s_dim, n_actions, noisy=noisy) 

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, init_log_alpha)
                
        self.update(rate=1.0)
    
    def forward(self, s):
        q1 = self.q1(s)
        q2 = self.q2(s)
        q1_target = self.q1_target(s)
        q2_target = self.q2_target(s)
        pi, log_pi = self.actor(s)
        log_alpha = self.log_alpha.view(-1,1)
        return q1, q2, q1_target, q2_target, pi, log_pi, log_alpha
    
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
        updateNet(self.q1_target, self.q1, rate)
        updateNet(self.q2_target, self.q2, rate)


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


class vision_actor_critic_Net(nn.Module):
    def __init__(self, s_dim, n_actions, n_heads, init_log_alpha=0.0, latent_dim=256, noisy=True):
        super().__init__()

        self.vision_net = vision_Net(latent_dim=latent_dim, noisy=noisy)
        self.actor_critic_net = discrete_actor_critic_Net(s_dim + latent_dim, n_actions, n_heads, 
                                                            init_log_alpha, noisy=noisy)
    
    def forward(self, inner_state, outer_state):
        observation = self.observe(inner_state, outer_state)
        output = self.actor_critic_net(observation)
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
        A, PA_s = self.actor_critic_net.sample_action(observation, explore=explore, rng=rng)
        return A, PA_s
    
    def update(self, rate=5e-3):
        self.actor_critic_net.update(rate)
    
    def get_alpha(self):
        return self.actor_critic_net.log_alpha.exp().item()


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
