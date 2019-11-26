import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Normal, Categorical

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def set_seed(n_seed):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)

def gaussian_likelihood(x, mu, log_std, EPS):
    likelihood = -0.5 * (((x-mu)/(torch.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    # print(x.shape)
    # print(mu.shape)
    # print(log_std.shape)
    return likelihood

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def atanh(y):
    return 0.5*(torch.log((1+y+1e-12)/(1-y+1e-12)))

###########################################################################
#
#                               Classes
#
###########################################################################

class Memory:
    def __init__(self, capacity = 50000, n_seed=0):
        self.capacity = capacity
        self.data = []        
        self.pointer = 0
        set_seed(n_seed)
    
    def store(self, event):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.pointer] = event
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample(self, batch_size):
        if batch_size < len(self.data):
            return random.sample(self.data, int(batch_size)) 
        else:
            return random.sample(self.data, len(self.data))

    def retrieve(self):
        return np.copy(self.data)
    
    def forget(self):
        self.data = []
        self.pointer = 0

    def empty(self):
        return len(self.data) == 0

#-------------------------------------------------------------
#
#    Value networks
#
#-------------------------------------------------------------
class v_valueNet(nn.Module):
    def __init__(self, input_dim, n_tasks=1, lr=3e-4, init_method='glorot'):
        super().__init__()        
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_tasks)  

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_)       

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)           
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

class q_valueNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, init_method='glorot'):
        super().__init__()        
        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_tasks)  

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)


class Q_tableNet(nn.Module):
    def __init__(self, task_param_dim, n_m_states, n_m_actions, hidden_dim=32, lr=3e-4, init_method='glorot'):
        super().__init__()
        self.l1 = nn.Linear(task_param_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, n_m_states*n_m_actions)
        self.n_m_states = n_m_states
        self.n_m_actions = n_m_actions

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l2.weight.data.uniform_(-3e-3, 3e-3)
            self.l2.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x).view(x.size(0), self.n_m_states, self.n_m_actions)        
        return(x)      

class conceptLossNet(nn.Module):
    def __init__(self, s_dim, n_m_states, hidden_dim=256, lr=3e-5, init_method='glorot'):
        super().__init__()        
        self.l11 = nn.Linear(s_dim, hidden_dim//2)
        self.l12 = nn.Linear(n_m_states, hidden_dim//2)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)  

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s, posterior):
        x = F.relu(self.l11(s))
        posterior = F.relu(self.l12(posterior))
        x = torch.cat([x, posterior], 1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

class rNet(nn.Module):
    def __init__(self, s_dim, n_m_actions, n_tasks=1, lr=3e-4, init_method='glorot'):
        super().__init__()      
        self.log_func = 'torch'
        self.std_lim_method = 'clamp'
        self.min_log_stdev = -6
        self.max_log_stdev = 4
        self.EPS_sigma = 1e-10
        self.n_tasks = n_tasks
        self.n_m_actions = n_m_actions

        self.l1 = nn.Linear(s_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l31 = nn.Linear(256, n_tasks*n_m_actions)
        self.l32 = nn.Linear(256, n_tasks*n_m_actions)  
        
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l31.weight.data.uniform_(-3e-3, 3e-3)
            self.l31.bias.data.uniform_(-3e-3, 3e-3)
            self.l32.weight.data.uniform_(-3e-3, 3e-3)
            self.l32.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.l31(x).view(-1, self.n_tasks, self.n_m_actions)
        log_stdev = self.l32(x).view(-1, self.n_tasks, self.n_m_actions)

        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)

        return m, log_stdev
    
    def sample(self, s):
        m, log_stdev = self(s)
        stdev = torch.exp(log_stdev)
        r = m + stdev*torch.randn_like(m)
        return r, m, log_stdev, stdev

    def sample_and_cross_llhood(self, s, T):
        r, m, log_stdev, stdev = self.sample(s)
        r = r.detach()[np.arange(s.size(0)), T, :].unsqueeze(1).unsqueeze(3)
        m = m[np.arange(s.size(0)), T, :].unsqueeze(0).unsqueeze(2)
        
        if self.log_func == 'self':
            log_stdev = log_stdev[np.arange(s.size(0)), T, :].unsqueeze(0).unsqueeze(2)
            cross_llhood = gaussian_likelihood(r, m, log_stdev, self.EPS_sigma)
        elif self.log_func == 'torch':
            stdev = stdev[np.arange(s.size(0)), T, :].unsqueeze(0).unsqueeze(2)
            cross_llhood = Normal(m, stdev).log_prob(r)        

        return cross_llhood
    
    def llhood(self, r, s, T, cross=False):
        m, log_stdev = self(s)        
        # z = torch.zeros(s.size(0),1).to(device)
        # o = torch.ones(s.size(0),1).to(device)
        
        m = m[np.arange(s.size(0)), T, :]

        if self.log_func == 'self':
            log_stdev = log_stdev[np.arange(s.size(0)), T, :]
            if not cross:
                llhood = gaussian_likelihood(r, m, log_stdev, self.EPS_sigma)
            else:
                llhood = gaussian_likelihood(r.unsqueeze(0).unsqueeze(2), m.unsqueeze(1).unsqueeze(3), log_stdev.unsqueeze(1).unsqueeze(3), self.EPS_sigma)
        elif self.log_func == 'torch':
            stdev = torch.exp(log_stdev)[np.arange(s.size(0)), T, :]
            if not cross:
                llhood = Normal(m, stdev).log_prob(r)
            else:
                llhood = Normal(m.unsqueeze(1).unsqueeze(3), stdev.unsqueeze(1).unsqueeze(3)).log_prob(r.unsqueeze(0).unsqueeze(2))

        # log_unnormalized_posterior = llhood + 0.2*Normal(o, 0.2*o).log_prob(stdev)

        assert torch.all(llhood==llhood), 'Invalid memb llhoods'
        if not cross:
            assert len(llhood.shape) == 2, 'Wrong size'
            assert llhood.size(1) == self.n_m_actions, 'Wrong size'
        else:
            assert len(llhood.shape) == 4, 'Wrong size'
            assert llhood.size(2) == self.n_m_actions, 'Wrong size'        
        # assert llhood.size(1) == 1, 'Wrong size'
        # return log_unnormalized_posterior
        return llhood

    # def llhood(self, s, r, t):
    #     m, log_stdev = self(s)
    #     stdev = torch.exp(log_stdev)
    #     m_task = m[np.arange(s.size(0)), t].view(-1,1)
    #     log_stdev_task = log_stdev[np.arange(s.size(0)), t].view(-1,1)
    #     stdev_task = stdev[np.arange(s.size(0)), t].view(-1,1)

    #     # assert stdev.size(1) == 1, 'Wrong size'
    #     r_repeated = r.view(1,1,r.size(0)).repeat(1,self.n_tasks,1)
    #     m_repeated = m.view(s.size(0),-1,1).repeat(1,1,r.size(0))
    #     if self.log_func == 'self':
    #         llhood = gaussian_likelihood(r, m_task, log_stdev_task, self.EPS_sigma)

    #         log_stdev_repeated = log_stdev.view(s.size(0),-1,1).repeat(1,1,r.size(0))
    #         cross_llhood = gaussian_likelihood(r_repeated, m_repeated, log_stdev_repeated, self.EPS_sigma)

    #     elif self.log_func == 'torch':
    #         llhood = Normal(m_task, stdev_task).log_prob(r)

    #         stdev_repeated = stdev.view(s.size(0),-1,1).repeat(1,1,r.size(0))
    #         cross_llhood = Normal(m_repeated, stdev_repeated).log_prob(r_repeated)
        

    #     assert torch.all(llhood==llhood), 'Invalid memb llhoods'

    #     assert llhood.size(1) == 1, 'Wrong size'

    #     return llhood, cross_llhood, m, log_stdev

class rewardNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, init_method='glorot'):
        super().__init__()      
        self.log_func = 'torch'
        self.std_lim_method = 'clamp'
        self.n_tasks = n_tasks
        
        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_tasks)
                
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 


        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s, a):
        x = torch.cat([s,a], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x    
    
class RNet(nn.Module):
    def __init__(self, n_m_states, n_tasks=1, lr=3e-4, latent_dim=10, hidden_dim=256, init_method='glorot'):
        super().__init__()      
        self.log_func = 'self'
        self.std_lim_method = 'clamp'
        self.min_log_stdev = -10
        self.max_log_stdev = 2
        self.EPS_sigma = 1e-6
        self.latent_dim = latent_dim
        self.n_m_states = n_m_states

        self.l1 = nn.Linear(latent_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l31 = nn.Linear(hidden_dim, n_tasks*n_m_states)
        self.l32 = nn.Linear(hidden_dim, n_tasks*n_m_states)  
        
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l31.weight.data.uniform_(-3e-3, 3e-3)
            self.l31.bias.data.uniform_(-3e-3, 3e-3)
            self.l32.weight.data.uniform_(-3e-3, 3e-3)
            self.l32.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim).float().cuda()
        x = F.relu(self.l1(z))
        x = F.relu(self.l2(x))
        m = self.l31(x)
        log_stdev = self.l32(x)

        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)

        return m.view(batch_size, -1, self.n_m_states), log_stdev.view(batch_size, -1, self.n_m_states)
    
    def llhood(self, r, t):
        m, log_stdev = self(r.size(0))
        m = m[:,t,:]
        log_stdev = log_stdev[:,t,:]
        stdev = torch.exp(log_stdev)

        assert stdev.size(1) == self.n_m_states, 'Wrong size'

        if self.log_func == 'self':
            llhood = gaussian_likelihood(r, m, log_stdev, self.EPS_sigma)
        elif self.log_func == 'torch':
            llhood = Normal(m, stdev).log_prob(r)
        assert torch.all(llhood==llhood), 'Invalid memb llhoods'

        assert llhood.size(1) == self.n_m_states, 'Wrong size'

        return llhood, m

class nextSNet(nn.Module):
    def __init__(self, s_dim, n_m_states, n_m_actions, n_tasks=1, lr=3e-5, init_method='glorot'):
        super().__init__()      
        self.log_func = 'torch'
        self.std_lim_method = 'clamp'
        self.min_log_stdev = -20
        self.max_log_stdev = 2
        self.EPS_sigma = 1e-10
        self.n_m_states = n_m_states

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, n_m_states*n_m_actions)  
        
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x)).view(x.size(0),x.size(1),-1,self.n_m_states)        
        # x = torch.exp(x - x.max(3, keepdim=True)[0])
        x = x / x.sum(3, keepdim=True)
        return x

#-------------------------------------------------------------
#
#    Tensor networks
#
#-------------------------------------------------------------
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

    def forward(self, input):
        return torch.einsum('ijk,jlk->ijl', input, self.weight) + self.bias.unsqueeze(0) 

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

    def forward(self, input):
        return torch.einsum('ik,jlk->ijl', input, self.weight) + self.bias.unsqueeze(0) 

    def single_output(self, input, label):
        weight = self.weight.data[label,:,:].view(self.out_features, self.in_features)
        bias = self.bias.data[label,:].view(self.out_features)
        output = input.matmul(weight.t()) + bias
        return output      

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) 

class v_parallel_valueNet(nn.Module):
    def __init__(self, s_dim, n_tasks=1, lr=3e-4, init_method='glorot'):
        super().__init__()        
        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 1)  

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_)       

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)           
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x).squeeze(2)
        return(x)

class q_parallel_valueNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, init_method='glorot'):
        super().__init__()        
        self.l1 = parallel_Linear_simple(n_tasks, s_dim+a_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 1)  

        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x).squeeze(2)
        return(x)

class mixtureConceptModel(nn.Module):
    def __init__(self, n_m_states, latent_dim, output_dim, min_log_stdev=-4, max_log_stdev=2, 
        llhood_samples=256, KL_samples=256, lr=3e-4, state_norm=False, min_c=25, init_method='glorot'):
        super().__init__()  
        self.s_dim = output_dim   
        self.n_m_states = n_m_states
        self.latent_dim = latent_dim
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev
        self.llhood_samples = llhood_samples
        self.KL_samples = KL_samples
        self.state_normalization = state_norm
        self.min_c = min_c
        self.EPS_sigma = 1e-8
        self.EPS_log_1_min_a2 = 1e-6
        self.std_lim_method = 'clamp' # 'squash' or 'clamp'
        self.log_lim_method = 'sum' # 'sum' or 'clamp'
        self.log_func = 'self' # 'torch' or 'self' , 'torch' in Hopper from 109 
        self.density_method = 'max' # 'mean' or 'max' 

        self.l1 = parallel_Linear(n_m_states, latent_dim, 256)
        self.l2 = parallel_Linear(n_m_states, 256, 256)
        self.l31 = parallel_Linear(n_m_states, 256, output_dim)
        self.l32 = parallel_Linear(n_m_states, 256, output_dim)
        
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l31.weight.data.uniform_(-3e-3, 3e-3)
            self.l32.weight.data.uniform_(-3e-3, 3e-3)
            self.l31.bias.data.uniform_(-3e-3, 3e-3)
            self.l32.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, batch_size):
        t = torch.randn(batch_size, 1, self.latent_dim).repeat(1,self.n_m_states,1).float().cuda()
        x = F.relu(self.l1(t))
        x = F.relu(self.l2(x))
        m = self.l31(x)
        if self.state_normalization:
            m = torch.tanh(m)
        assert torch.all(m==m), 'Invalid memb mean - sample'
        log_stdev = self.l32(x)
        assert torch.all(log_stdev==log_stdev), 'Invalid memb stdev - sample'
        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev

    def llhood(self, s):
        if self.state_normalization:
            s = torch.tanh(s)
        m, log_stdev = self(self.llhood_samples)
        stdev = log_stdev.exp()
        s_repeated = s.view(s.size(0),1,1,s.size(1)).repeat(1,self.llhood_samples,self.n_m_states,1)

        if self.log_func == 'self':
            llhoods = gaussian_likelihood(s_repeated, m, log_stdev, self.EPS_sigma).sum(3)
        elif self.log_func == 'torch':
            llhoods = Normal(m, stdev).log_prob(s_repeated).sum(3)
        assert torch.all(llhoods==llhoods), 'Invalid memb llhoods'

        if self.density_method == 'mean':
            llhoods = llhoods.mean(1)
        elif self.density_method == 'max':
            llhoods = llhoods.max(1)[0]
        elif self.density_method == 'soft':
            llhoods = 0.1*torch.logsumexp(llhoods/0.1, dim=1)

        # llhoods = ((Normal(m, stdev).log_prob(s_repeated)).sum(dim=3)).mean(dim=1)
        # assert torch.all(llhoods==llhoods), 'Invalid memb llhoods'
        llhoods = torch.clamp(llhoods, self.min_log_stdev*self.min_c, self.max_log_stdev)
        # assert not llhoods.ne(llhoods).any(), 'Told yah!'
        # llhoods[llhoods.ne(llhoods)]= self.min_log_stdev        
        return llhoods  

    def sample_m_state(self, s, explore=True):
        if self.state_normalization:
            s = torch.tanh(s)
        llhoods = self.llhood(s.view(1,-1)).view(-1)
        lmmbrship = torch.logsumexp(llhoods,0)
        log_posterior = (llhoods-lmmbrship)
        if explore:
            S = Categorical(logits=log_posterior).sample().item()
        else:
            posterior = torch.exp(log_posterior)
            tie_breaking_dist = torch.isclose(posterior, posterior.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().item()
            # S = log_posterior.argmax().item()
        return S
    
    def sample(self, batch_size):
        m, log_stdev = self(batch_size)
        stdev = log_stdev.exp()      
        s = m + stdev*torch.randn_like(m)        
        return s
    
    def sample_m_state_and_posterior(self, s, explore=True):
        llhoods = self.llhood(s)
        lmmbrship = torch.logsumexp(llhoods, 1, keepdim=True)        
        log_posterior = (llhoods-lmmbrship)
        posterior = torch.exp(log_posterior)
        if explore:
            S = Categorical(logits=log_posterior).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(posterior, posterior.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().cpu()
            # S = log_posterior.argmax(1).cpu()        
        return S, posterior
    
    # def posterior(self, s):
    #     llhoods = self.llhood(s)
    #     lmmbrship = torch.logsumexp(llhoods, 1, keepdim=True)
    #     post = torch.exp(llhoods - lmmbrship)
    #     return post
    
    # def KL_divergence(self):
    #     m, log_stdev = self(self.KL_samples)
    #     stdev = log_stdev.exp()      
    #     S = Categorical(torch.ones(self.n_m_states)/self.n_m_states).sample([self.KL_samples]).cpu()
    #     m_S = m[list(range(self.KL_samples)),S.tolist(),:]
    #     stdev_S = stdev[list(range(self.KL_samples)),S.tolist(),:]
    #     s = m_S + stdev_S*torch.randn_like(m_S)
    #     llhoods = self.llhood(s)
    #     lmmbrship = torch.logsumexp(llhoods, 1, keepdim=True)
    #     KL = (lmmbrship - llhoods).sum(dim=1).mean()
    #     return KL, lmmbrship
    
    # def KL_divergence_pairs(self):
    #     m, log_stdev = self(self.KL_samples)
    #     stdev = log_stdev.exp()     
    #     z = m + stdev*torch.randn_like(m)         
    #     llhoods = self.llhood(z.view(-1,self.s_dim)).view(-1,self.n_m_states,self.n_m_states)
    #     if self.state_normalization:
    #         s = torch.tanh(z)
    #         tanh_llhoods = - torch.log(torch.clamp(1 - s.pow(2), 1e-6, 1.0)).sum(dim=2).view(-1,1,self.n_m_states)
    #         llhoods += tanh_llhoods.repeat(1,self.n_m_states,1)
    #     sum_logs = llhoods.sum(dim=2)
    #     logs = self.n_m_states * llhoods[:,list(range(self.n_m_states)),list(range(self.n_m_states))]
    #     KL = (logs - sum_logs).mean()        
    #     return KL
    
    # def sample_and_KL_divergence(self):
    #     m, log_stdev = self(self.KL_samples)
    #     stdev = log_stdev.exp()      
    #     S = Categorical(torch.ones(self.n_m_states)/self.n_m_states).sample([self.KL_samples]).cpu()
    #     m_S = m[list(range(self.KL_samples)),S.tolist(),:]
    #     stdev_S = stdev[list(range(self.KL_samples)),S.tolist(),:]
    #     s = m_S + stdev_S*torch.randn_like(m_S)
    #     llhoods = self.llhood(s)
    #     lmmbrship = torch.logsumexp(llhoods, 1, keepdim=True)
    #     KL = (lmmbrship - llhoods).sum(dim=1).mean()
    #     return s, KL
    
    # def sample(self, batch_size):
    #     m, log_stdev = self(batch_size)
    #     stdev = log_stdev.exp()      
    #     S = Categorical(torch.ones(self.n_m_states)/self.n_m_states).sample([self.KL_samples]).cpu()
    #     m_S = m[list(range(self.KL_samples)),S.tolist(),:]
    #     stdev_S = stdev[list(range(self.KL_samples)),S.tolist(),:]
    #     s = m_S + stdev_S*torch.randn_like(m_S)        
    #     return s


class encoderConceptModel(nn.Module):
    def __init__(self, n_m_states, input_dim, n_tasks=1, hidden_dim=256, min_log_stdev=-4, max_log_stdev=2, lr=3e-4, min_c=25, init_method='glorot'):
        super().__init__()  
        self.s_dim = input_dim   
        self.n_m_states = n_m_states
        self.hidden_dim = hidden_dim
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev
        self.min_c = min_c
        self.EPS_sigma = 1e-8
        self.EPS_log_1_min_a2 = 1e-6
        self.std_lim_method = 'clamp' # 'squash' or 'clamp'
        self.log_lim_method = 'sum' # 'sum' or 'clamp'
        self.log_func = 'self' # 'torch' or 'self' , 'torch' in Hopper from 109 
        self.density_method = 'max' # 'mean' or 'max' 
        self.prior_n = 10000
        self.n_tasks = n_tasks

        self.l1 = parallel_Linear(n_m_states, self.s_dim, hidden_dim)
        self.l2 = parallel_Linear(n_m_states, hidden_dim, hidden_dim)
        self.l31 = parallel_Linear(n_m_states, hidden_dim, self.s_dim)
        self.l32 = parallel_Linear(n_m_states, hidden_dim, self.s_dim)

        self.prior = torch.ones(n_tasks, n_m_states).to(device)*self.prior_n/n_m_states
        
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l31.weight.data.uniform_(-3e-3, 3e-3)
            self.l32.weight.data.uniform_(-3e-3, 3e-3)
            self.l31.bias.data.uniform_(-3e-3, 3e-3)
            self.l32.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s):
        s = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_states,1)
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.l31(x)
        assert torch.all(m==m), 'Invalid memb mean - sample'
        log_stdev = self.l32(x)
        assert torch.all(log_stdev==log_stdev), 'Invalid memb stdev - sample'
        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)

        if self.log_func == 'self':
            llhoods = gaussian_likelihood(s, m, log_stdev, self.EPS_sigma).sum(2)
        elif self.log_func == 'torch':
            llhoods = Normal(m, torch.exp(log_stdev)).log_prob(s).sum(2)
        assert torch.all(llhoods==llhoods), 'Invalid memb llhoods'

        llhoods = torch.clamp(llhoods, self.min_log_stdev*self.min_c, self.max_log_stdev*self.min_c)
        
        return llhoods, m, log_stdev

    def sample_m_state(self, s, explore=True):
        llhoods = self(s.view(1,-1))[0].view(-1) + torch.log(self.prior.mean(0)/self.prior_n+1e-12)
        lmmbrship = torch.logsumexp(llhoods,0)
        log_posterior = (llhoods-lmmbrship)
        posterior = torch.exp(log_posterior)
        if explore:
            S = Categorical(logits=log_posterior).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(posterior, posterior.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().item()
            # S = log_posterior.argmax().item()
        return S, posterior
    
    def sample_m_state_and_posterior(self, s, explore=True):
        llhoods = self(s)[0] + torch.log(self.prior.mean(0)/self.prior_n+1e-20).view(1,-1)
        lmmbrship = torch.logsumexp(llhoods, 1, keepdim=True)        
        log_posterior = (llhoods-lmmbrship)
        posterior = torch.exp(log_posterior)
        if explore:
            S = Categorical(logits=log_posterior).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(posterior, posterior.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().cpu()
            # S = log_posterior.argmax(1).cpu()        
        return S, posterior, log_posterior
    
    def update_prior(self, S, T, P_S, method):
        if method == 'discrete':
            self.prior[T,S] += 1.0
        else:
            self.prior[T,:] += torch.FloatTensor(P_S.reshape(-1)).to(device).clone()
        self.prior[T,:] *= self.prior_n/(self.prior_n+1)

    # def update_prior(self, S, T):
    #     self.prior[T,S] += 1
    #     self.prior[T,:] *= self.prior_n/(self.prior_n+1)

class policyNet(nn.Module):
    def __init__(self, n_m_actions, input_dim, output_dim, min_log_stdev=-20, max_log_stdev=2, lr=3e-4, hidden_dim=256, 
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
        # self.l12 = parallel_Linear(n_m_actions, input_dim + self.latent_dim, hidden_dim)
        self.l21 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
        # self.l22 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
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

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # self.parameters_mean = list(self.l11.parameters()) + list(self.l21.parameters()) + list(self.l31.parameters())
        # self.optimizer_mean = optim.Adam(self.parameters_mean, lr=lr)
    
    def single_forward(self, s, A):
        if self.latent_dim > 0:
            t = torch.randn(1, self.latent_dim).float().cuda()
            s = torch.cat([s,t], 1)
        x1 = F.relu(self.l11.single_output(s, A))
        # x2 = F.relu(self.l12.single_output(s, A))
        x1 = F.relu(self.l21.single_output(x1, A))
        # x2 = F.relu(self.l22.single_output(x2, A))
        m = self.l31.single_output(x1, A)
        # log_stdev = self.l32.single_output(x2, A)
        log_stdev = self.l32.single_output(x1, A)
        log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev
    
    def sample_action(self, s, A, explore=True):
        m, log_stdev = self.single_forward(s, A)
        assert torch.all(m==m), 'Invalid mean action - sample'
        assert torch.all(log_stdev==log_stdev), 'Invalid action stdev - sample'
        if explore:
            u = m + log_stdev.exp()*torch.randn_like(m)
        else:
            u = m.clone()
        assert torch.all(u==u), 'Invalid free action - sample'
        a = torch.tanh(u).cpu()
        assert torch.all(a==a), 'Invalid constrained action - sample'      
          
        return a

    def calculate_mean(self, s):
        x = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        if self.latent_dim > 0:
            t = torch.randn(x.size(0), 1, self.latent_dim).repeat(1,self.n_m_actions,1).float().cuda()
            x = torch.cat([x,t], 2)
        x = F.relu(self.l11(x))
        x = F.relu(self.l21(x))
        m = self.l31(x).clone()
        return m
    
    def sample_action_and_llhood_pairs(self, s, A, explore=True):
        s = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(s)
        stdev = log_stdev.exp()
        if explore:
            u = m + stdev*torch.randn_like(m)
        else:
            u = m
        a = torch.tanh(u)

        selected_u = u[list(range(s.size(0))),A,:]
        selected_a = a[list(range(s.size(0))),A,:]

        assert len(selected_a.size()) == 2, 'Wrong size'
        assert selected_a.size(1) == self.a_dim, 'Wrong size'

        u_repeated = selected_u.view(s.size(0),1,self.a_dim).repeat(1,self.n_m_actions,1)
        a_repeated = selected_a.view(s.size(0),1,self.a_dim).repeat(1,self.n_m_actions,1)
        
        if self.log_func == 'self':
            llhoods = gaussian_likelihood(u_repeated, m, log_stdev, self.EPS_sigma)
        elif self.log_func == 'torch':
            llhoods = Normal(m, stdev).log_prob(u_repeated)

        if self.log_lim_method == 'clamp':
            llhoods -= torch.log(torch.clamp(1 - a_repeated.pow(2), self.EPS_log_1_min_a2, 1.0))    
        elif self.log_lim_method == 'sum':
            llhoods -= torch.log(1 - a_repeated.pow(2) + self.EPS_log_1_min_a2)

        llhoods = llhoods.sum(2)       

        # u_repeated = u.view(-1,1,self.n_m_actions,self.a_dim).repeat(1,self.n_m_actions,1,1)
        # a_repeated = a.view(-1,1,self.n_m_actions,self.a_dim).repeat(1,self.n_m_actions,1,1)
        # m_repeated = m.repeat(1,1,self.n_m_actions)   
        # stdev_repeated = stdev.repeat(1,1,self.n_m_actions)  
        # llhoods = (Normal(m_repeated, stdev_repeated + self.EPS_sigma).log_prob(u_repeated) - torch.log(torch.clamp(1 - a_repeated.pow(2), self.EPS_log_1_min_a2, 1.0)))
        # llhoods = Normal(m_repeated, stdev_repeated).log_prob(u_repeated) - torch.log(1 - a_repeated.pow(2))
        # llhoods = llhoods.view(-1, self.n_m_actions, self.n_m_actions, self.a_dim).sum(dim=3).transpose(1,2)
        # llhoods = torch.clamp(llhoods, self.min_log_stdev*self.min_c, self.max_log_stdev*1)
        
        return a, selected_a, llhoods
    
    def llhoods(self, s, a):
        s = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        a = a.view(a.size(0),1,a.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(s)
        stdev = log_stdev.exp()
        u = atanh(a)

        if self.log_func == 'self':
            llhoods = gaussian_likelihood(u, m, log_stdev, self.EPS_sigma)
        elif self.log_func == 'torch':
            llhoods = Normal(m, stdev).log_prob(u)

        if self.log_lim_method == 'clamp':
            llhoods -= torch.log(torch.clamp(1 - a.pow(2), self.EPS_log_1_min_a2, 1.0))    
        elif self.log_lim_method == 'sum':
            llhoods -= torch.log(1 - a.pow(2) + self.EPS_log_1_min_a2)

        llhoods = llhoods.sum(2)       

        return llhoods
    
    def sample_actions(self, s):
        s = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(s)
        stdev = log_stdev.exp()
        u = m + stdev*torch.randn_like(m)
        a = torch.tanh(u)
        return a
    
    def sample_actions_and_llhoods_for_all_skills(self, s, explore=True):
        s = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(s)
        stdev = log_stdev.exp()
        if explore:
            u = m + stdev*torch.randn_like(m)
        else:
            u = m
        a = torch.tanh(u)

        # u_repeated = selected_u.view(s.size(0),1,self.a_dim).repeat(1,self.n_m_actions,1)
        # a_repeated = selected_a.view(s.size(0),1,self.a_dim).repeat(1,self.n_m_actions,1)
        
        if self.log_func == 'self':
            llhoods = gaussian_likelihood(u.unsqueeze(1), m.unsqueeze(2), log_stdev.unsqueeze(2), self.EPS_sigma)
        elif self.log_func == 'torch':
            llhoods = Normal(m.unsqueeze(2), stdev.unsqueeze(2)).log_prob(u.unsqueeze(1))

        if self.log_lim_method == 'clamp':
            llhoods -= torch.log(torch.clamp(1 - a.unsqueeze(1).pow(2), self.EPS_log_1_min_a2, 1.0))    
        elif self.log_lim_method == 'sum':
            llhoods -= torch.log(1 - a.unsqueeze(1).pow(2) + self.EPS_log_1_min_a2)

        llhoods = llhoods.sum(3)   

        return a, llhoods

    def forward(self, s):
        x = s.clone()
        if self.latent_dim > 0:
            t = torch.randn(s.size(0), 1, self.latent_dim).repeat(1,self.n_m_actions,1).float().cuda()
            x = torch.cat([s,t], 2)
        x1 = F.relu(self.l11(x))
        # x2 = F.relu(self.l12(s))
        x1 = F.relu(self.l21(x1))
        # x2 = F.relu(self.l22(x2))
        m = self.l31(x1)
        # log_stdev = self.l32(x2)
        log_stdev = self.l32(x1)
        if self.std_lim_method == 'squash':
            log_stdev = 0.5 * (torch.tanh(log_stdev) + 1) * (self.max_log_stdev - self.min_log_stdev) + self.min_log_stdev
        elif self.std_lim_method == 'clamp':
            log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev

    # def sample_action_and_llhoods(self, s, A):
    #     s = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
    #     m, log_stdev = self(s)
    #     stdev = log_stdev.exp()
    #     m_A, stdev_A = m[list(range(s.size(0))),A,:], stdev[list(range(s.size(0))),A,:]        
    #     u = m_A + stdev_A*torch.randn_like(m_A)
    #     a = torch.tanh(u)
    #     u_repeated = u.view(u.size(0),1,u.size(1)).repeat(1,self.n_m_actions,1)        
    #     a_repeated = a.view(a.size(0),1,a.size(1)).repeat(1,self.n_m_actions,1)
    #     llhoods = (Normal(m, stdev).log_prob(u_repeated) - torch.log(torch.clamp(1 - a_repeated.pow(2), 1e-100, 1.0))).sum(dim=2)
    #     return a, llhoods 

    # def parallel_sample_action_and_llhood(self, s):
    #     m, log_stdev = self(s)
    #     stdev = log_stdev.exp()
    #     u = m + stdev*torch.randn_like(m)
    #     a = torch.tanh(u)
    #     llhood = (Normal(m, stdev).log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), 1e-100, 1.0))).sum(dim=2, keepdim=True)
    #     return a, llhood


class conditionalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, lr=3e-4, init_method='glorot'):
        super().__init__()
        hidden_dim_1 = int(input_dim//1.2)
        hidden_dim_2 = int(hidden_dim_1//1.2)
        self.l1 = nn.Linear(input_dim, hidden_dim_1)
        self.l2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.l3 = nn.Linear(hidden_dim_2, output_dim)
        
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)            
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

