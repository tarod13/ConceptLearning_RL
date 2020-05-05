import math
import random
import copy
import numpy as np
import heapq as hq

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Normal, Categorical

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Functions
##################################
def set_seed(n_seed):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)

def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def weights_init_noisy(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(m.bias, a=np.sqrt(5))

def weights_init_rnd(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        try:
            m.bias.data.zero_()
        except:
            pass

def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

# Classes
###################################
# Replay buffers
#------------------
class Memory:
    def __init__(self, capacity = 50000, n_seed=0):
        self.capacity = capacity
        self.data = []        
        self.pointer = 0
    
    def store(self, event):
        if self.len_data < self.capacity:
            self.data.append(None)
        self.data[self.pointer] = event
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample(self, batch_size):
        if batch_size < len(self.data):
            return random.sample(self.data, int(batch_size)) 
        else:
            return random.sample(self.data, self.len_data)

    def retrieve(self):
        return np.copy(self.data)
    
    def forget(self):
        self.data = []
        self.pointer = 0

    @property
    def empty(self):
        return len(self.data) == 0
    
    @property
    def len_data(self):
        return len(self.data)

# Custom neural layers
#------------------------
class parallel_Linear(nn.Module):
    def __init__(self, n_layers, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.weight = Parameter(torch.Tensor(n_layers, out_features, in_features))
        self.bias = Parameter(torch.Tensor(n_layers, out_features))        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, vars_=None):
        if vars_ is None:
            weight = self.weight
            bias = self.bias
        else:
            weight, bias = vars_
        return torch.einsum('ijk,jlk->ijl', input, weight) + bias.unsqueeze(0)

    def conditional(self, input, given):
        return torch.einsum('ik,lk->il', input, self.weight[given,:,:]) + self.bias[given,:].unsqueeze(0) 

    def single_output(self, input, label):
        weight = self.weight.data[label,:,:].view(self.out_features, self.in_features)
        bias = self.bias.data[label,:].view(self.out_features)
        output = input.matmul(weight.t()) + bias
        return output      

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) 

class parallel_Linear_simple(nn.Module):
    def __init__(self, n_layers, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.weight = Parameter(torch.Tensor(n_layers, out_features, in_features))
        self.bias = Parameter(torch.Tensor(n_layers, out_features))        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, vars_=None):
        if vars_ is None:
            weight = self.weight
            bias = self.bias
        else:
            weight, bias = vars_
        return torch.einsum('ik,jlk->ijl', input, weight) + bias.unsqueeze(0) 

    def single_output(self, input, label):
        weight = self.weight.data[label,:,:].view(self.out_features, self.in_features)
        bias = self.bias.data[label,:].view(self.out_features)
        output = input.matmul(weight.t()) + bias
        return output      

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Value networks
#------------------       
class v_net(nn.Module):
    def __init__(self, input_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks

        self.l1 = parallel_Linear_simple(n_tasks, input_dim, 256)        
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 1)

        self.apply(weights_init_)       
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        V = self.l3(x).squeeze(2)
        return V

class q_net(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks, lr=3e-4):
        super().__init__()      
        self.n_tasks = n_tasks 

        self.l1 = parallel_Linear_simple(n_tasks, s_dim+a_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 1)

        self.apply(weights_init_) 
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)   
    
    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        Q = self.l3(x).squeeze(2)
        return Q

class dueling_q_net(nn.Module):
    def __init__(self, s_dim, n_skills, n_tasks, lr=3e-4):
        super().__init__()  
        self.s_dim = s_dim
        self.n_skills = n_skills
        self.n_tasks = n_tasks   
        
        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.lV = parallel_Linear(n_tasks, 256, 1)
        self.lA = parallel_Linear(n_tasks, 256, n_skills)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV.weight, 0.01)
        self.lV.bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA.weight, 0.01)
        self.lA.bias.data.zero_()

        self.loss_func = nn.SmoothL1Loss() 
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)    
    
    def forward(self, s):        
        mu = F.relu(self.l1(s))
        mu = F.relu(self.l2(mu))
        V = self.lV(mu)        
        A = self.lA(mu)
        Q = V + A - A.mean(2, keepdim=True) 
        return Q

class noisy_dueling_q_net(nn.Module):
    def __init__(self, s_dim, n_skills, n_tasks, lr=3e-4):
        super().__init__()  
        self.s_dim = s_dim
        self.n_skills = n_skills
        self.n_tasks = n_tasks   
        
        self.l11 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l12 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l21 = parallel_Linear(n_tasks, 256, 256)
        self.l22 = parallel_Linear(n_tasks, 256, 256)
        self.lV1 = parallel_Linear(n_tasks, 256, 1)
        self.lV2 = parallel_Linear(n_tasks, 256, 1)
        self.lA1 = parallel_Linear(n_tasks, 256, n_skills)
        self.lA2 = parallel_Linear(n_tasks, 256, n_skills) 
        
        self.apply(weights_init_noisy)
        self.loss_func = nn.SmoothL1Loss() 
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)    
    
    def forward(self, s):        
        mu = self.l11(s)
        log_sigma = self.l12(s).clamp(-20.0,2.0)
        ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        mu = self.l21(x)
        log_sigma = self.l22(x).clamp(-20.0,2.0)
        ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        muV = self.lV1(x)
        log_sigmaV = self.lV2(x).clamp(-20.0,2.0)
        eiV = torch.randn(muV.size(0), self.n_tasks, 1).to(device)
        ejV = torch.randn(1, self.n_tasks, muV.size(2)).to(device)
        eijV = torch.sign(eiV)*torch.sign(ejV)*(eiV).abs()**0.5*(ejV).abs()**0.5
        V = muV + eijV*torch.exp(log_sigmaV)

        muA = self.lA1(x)
        log_sigmaA = self.lA2(x).clamp(-20.0,2.0)
        eiA = torch.randn(muA.size(0), self.n_tasks, 1).to(device)
        ejA = torch.randn(1, self.n_tasks, muA.size(2)).to(device)
        eijA = torch.sign(eiA)*torch.sign(ejA)*(eiA).abs()**0.5*(ejA).abs()**0.5
        A = muA + eijA*torch.exp(log_sigmaA)

        Q = V + A - A.mean(2, keepdim=True) #.view(-1, self.n_tasks, self.n_skills)
        return Q

# Policies
#---------------
class softmax_policy_net(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, n_skills)
        self.l41 = nn.Softmax(dim=2)
        self.l42 = nn.LogSoftmax(dim=2)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.l3.weight, 0.01)
        self.l3.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)
        
    def forward(self, s):    
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)    
        PA_sT = self.l41(x)
        log_PA_sT = self.l42(x)
        return PA_sT, log_PA_sT

class skill_net(nn.Module):
    def __init__(self, n_m_actions, input_dim, output_dim, min_log_stdev=-20, max_log_stdev=2, lr=3e-4, hidden_dim=256):
        super().__init__()   
        self.a_dim = output_dim
        self.n_m_actions = n_m_actions  
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev        
        self.eps = 1e-6 

        self.l11 = parallel_Linear(n_m_actions, input_dim, hidden_dim)
        self.l12 = parallel_Linear(n_m_actions, input_dim, hidden_dim)
        self.l21 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
        self.l22 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
        self.l31 = parallel_Linear(n_m_actions, hidden_dim, output_dim)
        self.l32 = parallel_Linear(n_m_actions, hidden_dim, output_dim)

        self.apply(weights_init_)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
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
        
        llhoods = Normal(m.unsqueeze(2), stdev.unsqueeze(2)).log_prob(u.unsqueeze(1))
        llhoods -= torch.log(1 - a.unsqueeze(1).pow(2) + self.eps)
        llhoods = llhoods.sum(3) #.clamp(self.min_log_stdev, self.max_log_stdev)   

        return a, llhoods

# Actor-critic modules
#------------------------
class discrete_AC_mixed(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, n_concepts, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        
        self.qe1 = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe1_target = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        self.qe2 = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe2_target = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        self.qi1 = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        self.qi1_target = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        self.qi2 = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        self.qi2_target = dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)

        self.actor = softmax_policy_net(n_skills, s_dim, n_tasks, lr=lr)        

        self.log_alpha = torch.zeros(n_tasks).to(device)
        self.log_alpha.requires_grad = True
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        self.log_Alpha = torch.zeros(n_concepts).to(device)
        self.log_Alpha.requires_grad = True
        self.Alpha = self.log_Alpha.exp()
        self.Alpha_optim = optim.Adam([self.log_Alpha], lr=lr, eps=1e-4)

        updateNet(self.qe1_target, self.qe1, 1.0)
        updateNet(self.qe2_target, self.qe2, 1.0)
        updateNet(self.qi1_target, self.qi1, 1.0)
        updateNet(self.qi2_target, self.qi2, 1.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s, T):
        qe1 = self.qe1(s)[np.arange(s.size(0)), T, :]
        qe1_target = self.qe1_target(s)[np.arange(s.size(0)), T, :]
        qe2 = self.qe2(s)[np.arange(s.size(0)), T, :]
        qe2_target = self.qe2_target(s)[np.arange(s.size(0)), T, :]
        qi1_exp = self.qi1(s)[np.arange(s.size(0)), T, :]
        qi1_exp_target = self.qi1_target(s)[np.arange(s.size(0)), T, :]
        qi2_exp = self.qi2(s)[np.arange(s.size(0)), T, :]
        qi2_exp_target = self.qi2_target(s)[np.arange(s.size(0)), T, :]
        pi, log_pi = self.actor(s)
        pi, log_pi = pi[np.arange(s.size(0)), T, :], log_pi[np.arange(s.size(0)), T, :]        
        alpha, log_alpha = self.alpha[T].view(-1,1), self.log_alpha[T].view(-1,1)
        Alpha, log_Alpha = self.Alpha, self.log_Alpha
        return qe1, qe1_target, qe2, qe2_target, qi1_exp, qi1_exp_target, qi2_exp, qi2_exp_target, pi, log_pi, alpha, log_alpha, Alpha, log_Alpha
    
    def sample_skill(self, s, task, explore=True, rng=None):
        PA_sT = self.actor(s.view(1,-1))[0].squeeze(0)[task,:].view(-1)
        if rng is None:
            if explore or np.random.rand() > 0.95:
                A = Categorical(probs=PA_sT).sample().item()
            else:
                tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max()).float()
                tie_breaking_dist /= tie_breaking_dist.sum()
                A = Categorical(probs=tie_breaking_dist).sample().item()  
        else:
            if explore or rng.rand() > 0.95:
                A = rng.choice(self.n_skills, p=PA_sT.detach().cpu().numpy())
            else:
                A = PA_sT.detach().cpu().argmax().item()
        return A
    
    def sample_skills(self, s, T, explore=True):
        PA_sT = self.actor(s)[0][np.arange(s.shape[0]), T, :]        
        if explore:
            A = Categorical(probs=PA_sT).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            A = Categorical(probs=tie_breaking_dist).sample().cpu()                  
        return A
    
    def update_targets(self, lr):
        updateNet(self.qe1_target, self.qe1, lr)
        updateNet(self.qe2_target, self.qe2, lr)

class discrete_AC(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        
        self.qe1 = noisy_dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe1_target = noisy_dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        self.qe2 = noisy_dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe2_target = noisy_dueling_q_net(s_dim, n_skills, n_tasks, lr=lr)
        
        self.actor = softmax_policy_net(n_skills, s_dim, n_tasks, lr=lr)        
        
        self.log_alpha = torch.zeros(n_tasks).to(device)
        self.log_alpha.requires_grad = True
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        updateNet(self.qe1_target, self.qe1, 1.0)
        updateNet(self.qe2_target, self.qe2, 1.0)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s, T):
        qe1_0 = self.qe1(s)
        qe1 = qe1_0[np.arange(s.size(0)), T, :]
        qe1_target = self.qe1_target(s)[np.arange(s.size(0)), T, :]
        qe2 = self.qe2(s)[np.arange(s.size(0)), T, :]
        qe2_target = self.qe2_target(s)[np.arange(s.size(0)), T, :]
        pi, log_pi = self.actor(s)
        pi, log_pi = pi[np.arange(s.size(0)), T, :], log_pi[np.arange(s.size(0)), T, :]        
        alpha, log_alpha = self.alpha[T].view(-1,1), self.log_alpha[T].view(-1,1)
        return qe1, qe1_target, qe2, qe2_target, pi, log_pi, alpha, log_alpha
    
    def sample_skill(self, s, task, explore=True, rng=None):
        PA_sT = self.actor(s.view(1,-1))[0].squeeze(0)[task,:].view(-1)
        if rng is None:
            if explore or np.random.rand() > 0.95:
                A = Categorical(probs=PA_sT).sample().item()
            else:
                tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max()).float()
                tie_breaking_dist /= tie_breaking_dist.sum()
                A = Categorical(probs=tie_breaking_dist).sample().item()  
        else:
            if explore or rng.rand() > 0.95:
                A = rng.choice(self.n_skills, p=PA_sT.detach().cpu().numpy())
            else:
                A = PA_sT.detach().cpu().argmax().item()
        return A
    
    def sample_skills(self, s, T, explore=True):
        PA_sT = self.actor(s)[0][np.arange(s.shape[0]), T, :]        
        if explore:
            A = Categorical(probs=PA_sT).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            A = Categorical(probs=tie_breaking_dist).sample().cpu()                  
        return A
    
    def update_targets(self, lr):
        updateNet(self.qe1_target, self.qe1, lr)
        updateNet(self.qe2_target, self.qe2, lr)
       
# RND structures, nets, and modules
#------------------------------------
class RewardForwardFilter(object):
    # https://github.com/jcwleo/random-network-distillation-pytorch/blob/e383fb95177c50bfdcd81b43e37c443c8cde1d94/utils.py#L51
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # https://github.com/jcwleo/random-network-distillation-pytorch/blob/e383fb95177c50bfdcd81b43e37c443c8cde1d94/utils.py#L51
    def __init__(self, n_tasks, s_dim, epsilon=1e-4, N=np.infty, init_mean=0.0):
        self.s_dim = s_dim
        self.n_tasks = n_tasks
        self.N = N

        self.mean = init_mean * torch.ones(n_tasks, s_dim).to(device)
        self.var = torch.ones(n_tasks, s_dim).to(device)
        self.sum_of_squared_diff = torch.ones(n_tasks, s_dim).to(device)
        self.count = epsilon * np.ones(n_tasks)

    def update(self, x, task, replace=False):
        batch_count = x.shape[0]
        if batch_count == 1:
            self.welford_online_update(x, task)
        else:
            batch_mean = x.mean(0).detach().view(1,-1)
            batch_var = x.var(0).detach().view(1,-1)        
            self.parallel_update_from_moments(batch_mean, batch_var, batch_count, task)

    def parallel_update_from_moments(self, batch_mean, batch_var, batch_count, task, replace=False):
        delta = batch_mean - self.mean[task, :].view(1,-1)
        new_count = self.count[task] + batch_count

        new_mean = self.mean[task, :].view(1,-1) + delta * batch_count / new_count
        m_a = self.var[task, :].view(1,-1) * (self.count[task])
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count[task] * batch_count / new_count
        new_var = M2 / new_count

        self.mean[task, :] = new_mean
        self.var[task, :] = new_var
        self.count[task] = min(self.N, new_count)
    
    def welford_online_update(self, x, task):
        delta = x.detach().view(1,-1) - self.mean[task, :].view(1,-1)
        new_count = self.count[task] + 1

        new_mean = self.mean[task, :].view(1,-1) + delta / new_count
        new_delta = x.detach().view(1,-1) - new_mean
        new_sum_of_squared_diff = self.sum_of_squared_diff[task, :].view(1,-1) + delta * new_delta
        new_var = self.var[task, :].view(1,-1) + (delta * new_delta - self.var[task, :].view(1,-1)) / new_count

        self.mean[task, :] = new_mean
        self.var[task, :] = new_var
        self.sum_of_squared_diff[task, :] = new_sum_of_squared_diff
        self.count[task] = min(self.N, new_count)

class RND_predictorNet(nn.Module):
    def __init__(self, s_dim, out_dim, n_tasks, lr=3e-4):
        super().__init__()  
        self.s_dim = s_dim
                
        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 512)
        self.l2 = parallel_Linear(n_tasks, 512, 512)
        self.l3 = parallel_Linear(n_tasks, 512, 512)
        self.l4 = parallel_Linear(n_tasks, 512, out_dim)
        
        self.apply(weights_init_rnd)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
   
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return(x)

class RND_targetNet(nn.Module):
    def __init__(self, s_dim, out_dim, n_tasks):
        super().__init__()  
        self.s_dim = s_dim
                
        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 512)
        self.l2 = parallel_Linear(n_tasks, 512, out_dim)
        
        self.apply(weights_init_rnd)

        for param in self.parameters():
            param.requires_grad = False       
        
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = self.l2(x)
        return(x)

class RND_module(nn.Module):
    def __init__(self, s_dim, n_tasks, gamma_I=0.99, max_dim_norm=31, out_dim=512, lr=1e-4, alpha=5e-2):
        super().__init__() 
        self.s_dim = s_dim
        self.out_dim = out_dim
        self.max_dim_norm = max_dim_norm
        self.alpha = alpha
        self.n_tasks = n_tasks
        self.gamma_I = gamma_I
        
        self.target = RND_targetNet(s_dim, out_dim, n_tasks)
        self.predictor = RND_predictorNet(s_dim, out_dim, n_tasks, lr=lr)
        self.predictor_frozen = RND_predictorNet(s_dim, out_dim, n_tasks, lr=lr)
        
        self.obs_rms = RunningMeanStd(n_tasks, s_dim)
        self.f_rms = RunningMeanStd(n_tasks, 1)
        self.rff_rms_int = RunningMeanStd(n_tasks, 1)
        self.rff_int = RewardForwardFilter(self.gamma_I)

        updateNet(self.predictor_frozen, self.predictor, 1.0)

    def normalize_obs(self, s, t):
        s_normalized = s.detach().clone()
        s_normalized[:,:self.max_dim_norm] = (s[:,:self.max_dim_norm] - self.obs_rms.mean[t,:self.max_dim_norm].view(1,-1)) / self.obs_rms.var[t,:self.max_dim_norm].view(1,-1)**0.5
        s_normalized[:,self.max_dim_norm:] = 2.0 * (s_normalized[:,self.max_dim_norm:] - 0.5)
        s_normalized = s_normalized.clamp(-5.0,5.0)
        return s_normalized    
    
    def forward(self, s, t, update_rms=True):
        s_normalized = self.normalize_obs(s, t)
        target = self.target(s_normalized)
        prediction = self.predictor(s_normalized)
        error = ((prediction - target.detach())[:, t, :])**2
        return error

    def update_obs_rms(self, s, t):
        self.obs_rms.update(s.detach(), t)
    
    def update_q_rms(self, q, t):
        self.q_rms.update(q.detach(), t)
    
    def reset_discounted_reward(self):
        self.discounted_reward = RewardForwardFilter(self.gamma_I) 
    
    def novelty_ratios(self, s, T):
        s_normalized = self.normalize_obs(s, T)
        target = self.target(s_normalized)
        prediction = self.predictor(s_normalized)
        prediction_0 = self.predictor_frozen(s_normalized)
        error = (((prediction - target.detach())[:,T,:])**2).sum(1, keepdim=True).detach()
        error_0 = (((prediction_0 - target.detach())[:,T,:])**2).sum(1, keepdim=True).detach()
        valid_error = (error_0 > error).float()
        self.f_rms.update(error_0, T)
        log_novelty_ratios = - torch.log(error + 1e-10)        
        log_novelty_ratios = log_novelty_ratios + torch.log(error_0 + 1e-10) * valid_error
        log_novelty_ratios = log_novelty_ratios + torch.log(self.f_rms.mean.view(1,-1).detach() + 1e-10) * (1-valid_error)        
        return log_novelty_ratios.clamp(0.0, np.infty)

# Classifiers
#--------------
class classifier_net(nn.Module):
    def __init__(self, n_concepts, s_dim, n_skills, n_tasks=1, lr=3e-4):
        super().__init__()  
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_concepts = n_concepts          
        
        self.l1 = nn.Linear(s_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_concepts)
        self.l4 = nn.Softmax(dim=1) # TODO: add logsoftmax net
        
        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, s):                 
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        PS_s = self.l4(x)
        log_PS_s = x - torch.logsumexp(x, dim=1, keepdim=True)
        return PS_s, log_PS_s
      
    def sample_concept(self, s, explore=True):
        PS_s = self.classifier(s.view(1,-1))[0].view(-1)
        if explore:
            S = Categorical(probs=PS_s).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(PS_s, PS_s.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().item()            
        return S

    def sample_concepts(self, s, explore=True, vars_=None):
        PS_s = self.classifier(s)[0]
        if explore:
            S = Categorical(probs=PS_s).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PS_s, PS_s.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            S = Categorical(probs=tie_breaking_dist).sample().cpu()   
        return S