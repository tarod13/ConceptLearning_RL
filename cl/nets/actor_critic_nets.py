import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.nn.parameter import Parameter
from torch.optim import Adam

from policy_nets import *
from q_nets import *
from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")



class actor_critic_Net(nn.Module):
    def __init__(self, s_dim, a_dim, noisy, lr=3e-4, lr_alpha=3e-4):
        super().__init__()   

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.q1 = q_Net(s_dim, a_dim, noisy, lr)        
        self.q1_target = q_Net(s_dim, a_dim, noisy, lr)
        self.q2 = q_Net(s_dim, a_dim, noisy, lr)        
        self.q2_target = q_Net(s_dim, a_dim, noisy, lr)
    
        self.actor = actor_Net(s_dim, a_dim, noisy, lr=lr)

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, 0.0)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
                
        self.update(rate=1.0)
    
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
        action = self.actor.sample_action(s.view(1,-1), explore)
        return action
    
    def update(self, rate=5e-3):
        updateNet(self.q1_target, self.q1, rate)
        updateNet(self.q2_target, self.q2, rate)



class discrete_vision_actor_critic_Net(nn.Module):
    def __init__(self, s_dim, n_actions, latent_dim, n_heads=8, init_log_alpha=0.0, 
                    parallel=True, lr=1e-4, lr_alpha=1e-4, lr_actor=1e-4):
        super().__init__()   

        self.s_dim = s_dim
        self.n_actions = n_actions     
        self._parallel = parallel    
        
        self.q = vision_multihead_dueling_q_Net(s_dim, latent_dim, n_actions, n_heads, lr)        
        self.q_target = vision_multihead_dueling_q_Net(s_dim, latent_dim, n_actions, n_heads, lr)
        self.update(rate=1.0)
        
        self.actor = vision_softmax_policy_Net(s_dim, latent_dim, n_actions, noisy=False, lr=lr_alpha) 

        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, init_log_alpha)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
    
    def forward(self):
        pass

    def evaluate_critic(self, inner_state, outer_state, next_inner_state, next_outer_state):
        q = self.q(inner_state, outer_state)
        next_q = self.q_target(next_inner_state, next_outer_state)
        next_pi, next_log_pi = self.actor(next_inner_state, next_outer_state)
        log_alpha = self.log_alpha.view(-1,1)
        return q, next_q, next_pi, next_log_pi, log_alpha
    
    def evaluate_actor(self, inner_state, outer_state):
        q = self.q(inner_state, outer_state)            
        pi, log_pi = self.actor(inner_state, outer_state)
        return q, pi, log_pi
    
    def sample_action(self, inner_state, outer_state, explore=True):
        PA_s = self.actor(inner_state.view(1,-1), outer_state.unsqueeze(0))[0].squeeze(0).view(-1)
        assert torch.all(PA_s == PA_s), 'Boom. Capoot.'
        if explore:
            A = Categorical(probs=PA_s).sample().item()
        else:
            tie_breaking_dist = torch.isclose(PA_s, PA_s.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().item()  
        return A, PA_s.detach().cpu().numpy()
    
    def update(self, rate=5e-3):
        updateNet(self.q_target, self.q, rate)
    
    def get_alpha(self):
        return self.log_alpha.exp().item()
