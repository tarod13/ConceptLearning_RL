import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.optim import Adam

from custom_layers import parallel_Linear, Linear_noisy
from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# Discrete action space
#-------------------------------------------------
class softmax_policy_Net(nn.Module):
    def __init__(self, s_dim, n_actions, noisy=False, lr=3e-4):
        super().__init__()
        
        self.s_dim = s_dim   
        self.n_actions = n_actions 

        if noisy:
            layer = Linear_noisy
        else:
            layer = nn.Linear

        self.logits_layer = layer(256, n_actions)
        self.logit_pipe = nn.Sequential(
            layer(s_dim, 256),
            nn.ReLU(),
            layer(256, 256),
            nn.ReLU(),
            self.logits_layer            
        )        
        
        if not noisy:
            self.logit_pipe.apply(weights_init_rnd)
            torch.nn.init.orthogonal_(self.logits_layer.weight, 0.01)
            self.logits_layer.bias.data.zero_()
        else:
            torch.nn.init.orthogonal_(self.logits_layer.mean_weight, 0.01)
            self.logits_layer.mean_bias.data.zero_()
            
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, s):    
        logits = self.logit_pipe(s) 
        PA_s = nn.Softmax(dim=1)(logits)
        log_PA_s = nn.LogSoftmax(dim=1)(logits)
        return PA_s, log_PA_s


class vision_softmax_policy_Net(softmax_policy_Net):
    def __init__(self, s_dim, latent_dim, n_actions, noisy=True, lr=1e-4):
        super().__init__(s_dim + latent_dim, n_actions, noisy)        
        self.vision_net = vision_Net(latent_dim=latent_dim, noisy=noisy)
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, inner_state, outer_state):    
        features = self.vision_net(outer_state)
        state = torch.cat([inner_state, features], dim=1)
        logits = self.logit_pipe(state) 
        PA_s = nn.Softmax(dim=1)(logits)
        log_PA_s = nn.LogSoftmax(dim=1)(logits)
        return PA_s, log_PA_s


# Continuous action space
#-------------------------------------------------
class actor_Net(nn.Module):
    def __init__(self, s_dim, a_dim, min_std=1e-20, max_std=1e2, noisy=False, lr=3e-4):
        super().__init__()   
        self.min_std = min_std
        self.max_std = max_std
        
        if noisy:
            layer = Linear_noisy
        else:
            layer = nn.Linear

        self.l1 = layer(s_dim, 256)
        self.l2 = layer(256, 256)
        self.l_mean = layer(256, a_dim)
        self.l_logstd = layer(256, a_dim)
        
        if not noisy:
            self.apply(weights_init_rnd)
        
        self.optimizer = Adam(self.parameters(), lr=lr)
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.l_mean(x)
        log_std = self.l_logstd(x)
        std = log_std.exp()
        std = torch.clamp(std.abs(), self.min_std, self.max_std)
        return m, std

    def simulate(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.l_mean(x)
        a = torch.tanh(m)        
        return a

    def sample_action(self, s, explore=True):
        with torch.no_grad():
            m, std = self(s)
            if explore:
                u = m + std*torch.randn_like(m)
            else:
                u = m
            a = torch.tanh(u).squeeze(0).cpu().numpy()        
            return a

    def sample_actions_and_llhoods(self, s, explore=True):
        m, std = self(s)
        if explore:
            u = m + std*torch.randn_like(m)
        else:
            u = m
        a = torch.tanh(u)
        
        llhoods = Normal(m, std.abs()).log_prob(u)
        llhoods -= torch.log(1 - a.pow(2) + 1e-6)
        llhoods = llhoods.sum(1, keepdim=True)
        return a, llhoods
    
    def llhoods(self, s, a):
        m, std = self(s)
        llhoods = Normal(m, std.abs()).log_prob(u)
        llhoods -= torch.log(1 - a.pow(2) + 1e-6)
        llhoods = llhoods.sum(1, keepdim=True)
        return llhoods


class s_Net(nn.Module):
    def __init__(self, n_m_actions, input_dim, output_dim, min_log_stdev=-20, max_log_stdev=2, hidden_dim=256, 
        latent_dim=0, min_c=2, init_method='glorot'):
        super().__init__()   
        self.a_dim = output_dim
        self.n_m_actions = n_m_actions  
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev
        self.latent_dim = latent_dim
        self.min_c = min_c
        self.EPS_sigma = 1e-8
        self.EPS_log_1_min_a2 = 1e-6
        self.std_lim_method = 'clamp' # 'squash' or 'clamp'
        self.log_lim_method = 'sum' # 'sum' or 'clamp'
        self.log_func = 'torch' # 'torch' or 'self'

        self.l11 = parallel_Linear(n_m_actions, input_dim + self.latent_dim, hidden_dim)
        self.l12 = parallel_Linear(n_m_actions, input_dim + self.latent_dim, hidden_dim)
        self.l21 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
        self.l22 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
        self.l31 = parallel_Linear(n_m_actions, hidden_dim, output_dim)
        self.l32 = parallel_Linear(n_m_actions, hidden_dim, output_dim)

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l31.weight.data.uniform_(-3e-3, 3e-3)
            self.l32.weight.data.uniform_(-3e-3, 3e-3)
            self.l31.bias.data.uniform_(-3e-3, 3e-3)
            self.l32.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_)
    
    def conditional(self, s, A):
        x = s.clone().view(1,s.size(0))

        mu = self.l11.conditional(x, A)
        log_sigma = self.l12.conditional(x, A).clamp(-20.0,4.0)
        ei = torch.randn(mu.size(0), 1).to(device)
        ej = torch.randn(1, mu.size(1)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        mu = self.l21.conditional(x, A)
        log_sigma = self.l22.conditional(x, A).clamp(-20.0,4.0)
        ei = torch.randn(mu.size(0), 1).to(device)
        ej = torch.randn(1, mu.size(1)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))
        
        m = self.l31.conditional(x, A)
        log_stdev = self.l32.conditional(x, A)
        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev
    
    def sample_action(self, s, A, explore=True):
        m, log_stdev = self.conditional(s, A)
        stdev = log_stdev.exp()
        if explore:
            u = m + stdev*torch.randn_like(m)
        else:
            u = m
        a = torch.tanh(u).squeeze(0).cpu().numpy()        
        return a
    
    def forward(self, s):
        mu = self.l11(s)
        log_sigma = self.l12(s).clamp(-20.0,4.0)
        ei = torch.randn(mu.size(0), self.n_m_actions, 1).to(device)
        ej = torch.randn(1, self.n_m_actions, mu.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        mu = self.l21(x)
        log_sigma = self.l22(x).clamp(-20.0,4.0)
        ei = torch.randn(mu.size(0), self.n_m_actions, 1).to(device)
        ej = torch.randn(1, self.n_m_actions, mu.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        m = self.l31(x)
        log_stdev = self.l32(x)
        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev
   
    def sample_actions_and_llhoods_for_all_skills(self, s, explore=True):
        x = s.clone().view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(x)
        stdev = log_stdev.exp()
        if explore:
            u = m + stdev*torch.randn_like(m)
        else:
            u = m
        a = torch.tanh(u)
        
        if self.log_func == 'self':
            llhoods = gaussian_likelihood(u.unsqueeze(1), m.unsqueeze(2), log_stdev.unsqueeze(2), self.EPS_sigma)
        elif self.log_func == 'torch':
            llhoods = Normal(m.unsqueeze(2), stdev.unsqueeze(2)).log_prob(u.unsqueeze(1))

        if self.log_lim_method == 'clamp':
            llhoods -= torch.log(torch.clamp(1 - a.unsqueeze(1).pow(2), self.EPS_log_1_min_a2, 1.0))    
        elif self.log_lim_method == 'sum':
            llhoods -= torch.log(1 - a.unsqueeze(1).pow(2) + self.EPS_log_1_min_a2)

        llhoods = llhoods.sum(3) #.clamp(self.min_log_stdev, self.max_log_stdev)   

        return a, llhoods