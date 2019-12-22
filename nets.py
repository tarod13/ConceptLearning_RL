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
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple) or isinstance(m, AutoregressiveLinear):
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
        self.min_log_stdev = -4
        self.max_log_stdev = 4
        self.EPS_sigma = 1e-10
        self.n_tasks = n_tasks
        self.n_m_actions = n_m_actions

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l31 = parallel_Linear(n_tasks, 256, n_m_actions)
        self.l32 = parallel_Linear(n_tasks, 256, n_m_actions)  
        
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
        m = self.l31(x) #.view(-1, self.n_tasks, self.n_m_actions)
        log_stdev = self.l32(x) #.view(-1, self.n_tasks, self.n_m_actions)

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

    # def sample_and_cross_llhood(self, s, T):
    #     r, m, log_stdev, stdev = self.sample(s)
    #     r = r.detach()[np.arange(s.size(0)), T, :].unsqueeze(1).unsqueeze(3)
    #     m = m[np.arange(s.size(0)), T, :].unsqueeze(0).unsqueeze(2)
        
    #     if self.log_func == 'self':
    #         log_stdev = log_stdev[np.arange(s.size(0)), T, :].unsqueeze(0).unsqueeze(2)
    #         cross_llhood = gaussian_likelihood(r, m, log_stdev, self.EPS_sigma)
    #     elif self.log_func == 'torch':
    #         stdev = stdev[np.arange(s.size(0)), T, :].unsqueeze(0).unsqueeze(2)
    #         cross_llhood = Normal(m, stdev).log_prob(r)        

    #     return cross_llhood
    
    def llhood(self, r, s, T, cross=False):
        m, log_stdev = self(s)        
        # z = torch.zeros(s.size(0),1).to(device)
        # o = torch.ones(s.size(0),1).to(device)
        
        m = m[np.arange(s.size(0)), T, :]
        log_stdev = log_stdev[np.arange(s.size(0)), T, :]

        if self.log_func == 'self':            
            if not cross:
                llhood = gaussian_likelihood(r, m, log_stdev, self.EPS_sigma)
            else:
                llhood = gaussian_likelihood(r.unsqueeze(0).unsqueeze(2), m.unsqueeze(1).unsqueeze(3), log_stdev.unsqueeze(1).unsqueeze(3), self.EPS_sigma)
        elif self.log_func == 'torch':
            stdev = torch.exp(log_stdev)
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
        # print("s.shape")
        # print(s.shape)
        # print("a.shape")
        # print(a.shape)
        x = torch.cat([s.clone(),a.clone()], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x  

class transitionNet(nn.Module):
    def __init__(self, s_dim, a_dim, lr=3e-4, init_method='glorot'):
        super().__init__()      
        
        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, s_dim)
                
        self.init_method = init_method
        if self.init_method == 'uniform':
            self.l3.weight.data.uniform_(-3e-3, 3e-3)
            self.l3.bias.data.uniform_(-3e-3, 3e-3)
        elif self.init_method == 'glorot':
            self.apply(weights_init_) 


        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s, a):
        x = torch.cat([s.clone(),a.clone()], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class SimPLe_encoderNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, distribution_type='discrete', min_T=5e-1, T_0=1.0, annealing_steps=1e5):
        super().__init__()    

        self.distribution_type = distribution_type
        self.min_log_stdev = -4
        self.max_log_stdev = 4  
        self.out_dim = s_dim//4+1
        if distribution_type == 'discrete':
            self.T = T_0
            self.min_T = min_T
            self.decrease_rate = (min_T/T_0)**(1.0/annealing_steps)
            self.annealing_steps = annealing_steps
        
        self.l1 = parallel_Linear_simple(n_tasks, 2*s_dim+a_dim, s_dim)
        self.l2 = parallel_Linear(n_tasks, s_dim, s_dim//2+1)
        self.l31 = parallel_Linear(n_tasks, s_dim//2+1, self.out_dim)
        if distribution_type != 'discrete':
            self.l32 = parallel_Linear(n_tasks, s_dim//2+1, self.out_dim)

        self.bn = nn.BatchNorm1d(n_tasks)
        self.d = nn.Dropout(0.2)
                
        self.apply(weights_init_)

    def forward(self, s, a, ns):
        if self.distribution_type == 'discrete':
            self.T = np.max([self.decrease_rate * self.T, self.min_T])

            x = torch.cat([s,a,ns],1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            mean = torch.sigmoid(self.l31(x))
            mean = mean / mean.sum(2, keepdim=True)
            
            u = torch.rand_like(mean)
            g = -torch.log((-torch.log(u+1e-10)).clamp(1e-10,1.0))
            y = torch.exp((torch.log(mean+1e-10) + g) / self.T)
            y = y / y.sum(2, keepdim=True)
            r = torch.rand_like(y)
            z = (y > r).float() + y - y.detach()

            log_stdev = mean
            
        else:
            x = torch.cat([s,a,ns],1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            mean = self.l31(x)
            log_stdev = self.l32(x)
            z = mean + torch.exp(log_stdev) * torch.randn_like(mean) #) * self.max_log_stdev
        
        return z, mean, log_stdev

# class SimPLe_encoderNet(nn.Module):
#     def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4):
#         super().__init__()    

#         self.min_log_stdev = -4
#         self.max_log_stdev = 4  
        
#         self.l1 = parallel_Linear_simple(n_tasks, 2*s_dim+a_dim, s_dim)
#         self.l2 = parallel_Linear(n_tasks, s_dim, s_dim//2+1)
#         self.l3 = parallel_Linear(n_tasks, s_dim//2+1, s_dim//4+1)
#         # self.l32 = parallel_Linear(n_tasks, s_dim//2+1, s_dim//4+1)

#         self.d = nn.Dropout(0.2)
                
#         self.apply(weights_init_)

#     def forward(self, s, a, ns):
#         x = torch.cat([s,a,ns],1)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.l31(x)
#         xn = x + torch.randn_like(x)
#         x1 = (1.2*torch.sigmoid(xn)-0.1).clamp(0.0,1.0)
#         x2 = xn < 0.0
#         x2 += x1 - x1.detach()

#         r = np.random.rand()
#         if r > 0.5:
#             xd = x1
#         else:
#             xd = x2 
#         xd = self.d(xd)

#         return xd  

# class SimPLe_decoderNet(nn.Module):
#     def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4):
#         super().__init__()      
        
#         self.l11 = parallel_Linear_simple(n_tasks, s_dim-40, (5*(s_dim-40))//6+1)
#         self.l12 = parallel_Linear_simple(n_tasks, 20, 10)
#         self.l13 = parallel_Linear_simple(n_tasks, 20, 10)
#         self.l21 = parallel_Linear(n_tasks, (5*(s_dim-40))//6+1, (3*(s_dim-40))//4+1)
#         self.l22 = parallel_Linear(n_tasks, 10, 5)
#         self.l23 = parallel_Linear(n_tasks, 10, 5)
#         self.l3 = parallel_Linear(n_tasks, (3*(s_dim-40))//4+11, s_dim//2+1)
#         self.l4 = parallel_Linear(n_tasks, s_dim//2+1+a_dim+s_dim//4+1, (3*s_dim)//4+1)
#         self.l5 = parallel_Linear(n_tasks, (3*s_dim)//4+1+a_dim, (5*s_dim)//6+1)
#         self.l61 = parallel_Linear(n_tasks, (5*s_dim)//6+1+a_dim, s_dim)
#         self.l62 = parallel_Linear(n_tasks, (5*s_dim)//6+1+a_dim, 1)
                
#         self.apply(weights_init_)
    
#     def forward(self, s, a, z, min_, max_):       
#         x1 = F.relu(self.l11(s[:,:-40]))
#         x2 = F.relu(self.l12(s[:,-40:-20]))
#         x3 = F.relu(self.l13(s[:,-20:]))
#         x1 = F.relu(self.l21(x1))
#         x2 = F.relu(self.l22(x2))
#         x3 = F.relu(self.l23(x3))
#         x = torch.cat([x1,x2,x3y],2)
#         x = F.relu(self.l3(x))
#         x = torch.cat([x,z,a],2)
#         x = F.relu(self.l4(x))
#         x = torch.cat([x,a],2)
#         x = F.relu(self.l5(x))
#         x = torch.cat([x,a],2)
#         # ns = 0.5*(torch.tanh(self.l61(x))+1) * (max_ - min_).view(1,1,-1).abs() + min_.view(1,1,-1)
#         ns = self.l61(x)
#         r = self.l62(x)
#         return ns, r

# class SimPLe_decoderNet(nn.Module):
#     def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4):
#         super().__init__()      
        
#         self.l1 = parallel_Linear_simple(n_tasks, s_dim, (5*s_dim)//6+1)
#         self.l2 = parallel_Linear(n_tasks, (5*s_dim)//6+1, (3*s_dim)//4+1)
#         self.l3 = parallel_Linear(n_tasks, (3*s_dim)//4+1, s_dim//2+1)
#         self.l4 = parallel_Linear(n_tasks, s_dim//2+1+a_dim+s_dim//4+1, (3*s_dim)//4+1)
#         self.l5 = parallel_Linear(n_tasks, (3*s_dim)//4+1+a_dim, (5*s_dim)//6+1)
#         self.l61 = parallel_Linear(n_tasks, (5*s_dim)//6+1+a_dim, s_dim)
#         self.l62 = parallel_Linear(n_tasks, (5*s_dim)//6+1+a_dim, 1)
                
#         self.apply(weights_init_)
    
#     def forward(self, s, a, z, min_, max_):       
#         x = F.relu(self.l1(s))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = torch.cat([x,z,a],2)
#         x = F.relu(self.l4(x))
#         x = torch.cat([x,a],2)
#         x = F.relu(self.l5(x))
#         x = torch.cat([x,a],2)
#         # ns = 0.5*(torch.tanh(self.l61(x))+1) * (max_ - min_).view(1,1,-1).abs() + min_.view(1,1,-1)
#         ns = self.l61(x)
#         r = self.l62(x)
#         return ns, r

class SimPLe_decoderNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, latent_dim=0):
        super().__init__()      
        dim1 = (5*s_dim)//6+1
        dim2 = (3*s_dim)//4+1
        dim3 = (1*s_dim)//2+1
        dim4 = (1*s_dim)//4+1

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, dim1)
        self.l3 = parallel_Linear(n_tasks, dim1, dim2)
        self.l5 = parallel_Linear(n_tasks, dim2, dim3)
        self.l7 = parallel_Linear(n_tasks, dim3, dim4)
        self.l8 = parallel_Linear(n_tasks, dim4+a_dim+latent_dim, dim3)
        self.l6 = parallel_Linear(n_tasks, dim3+a_dim, dim2)
        self.l4 = parallel_Linear(n_tasks, dim2+a_dim, dim1)
        self.l2 = parallel_Linear(n_tasks, dim1+a_dim, s_dim)
        
        # Skip - connection layers
        self.l02 = parallel_Linear_simple(n_tasks, s_dim, s_dim)
        self.l14 = parallel_Linear(n_tasks, dim1, dim1)
        self.l36 = parallel_Linear(n_tasks, dim2, dim2)
        self.l58 = parallel_Linear(n_tasks, dim3, dim3)

        # self.l_latent = parallel_Linear(n_tasks, latent_dim, dim2)
        self.l_reward = parallel_Linear(n_tasks, s_dim+a_dim+dim4+latent_dim, 1)

        # self.l1 = parallel_Linear(n_tasks, s_dim+a_dim+latent_dim, 256)
        # self.l2 = parallel_Linear(n_tasks, 256, 256)
        # self.l3 = parallel_Linear(n_tasks, 256, s_dim)

        # self.l_reward = parallel_Linear(n_tasks, 256, 1)

        self.bn = nn.BatchNorm1d(n_tasks)
        self.d = nn.Dropout(0.2)
                
        self.apply(weights_init_)
    
    def forward(self, s, a, z, min_, max_):
        x1 = F.relu(self.bn(self.l1(s)))
        x3 = F.relu(self.bn(self.l3(x1)))
        x5 = F.relu(self.bn(self.l5(x3)))
        x_abstract = F.relu(self.bn(self.l7(x5)))
        x7 = torch.cat([x_abstract,z,a],2)
        x8 = F.relu(self.bn(self.l8(x7) + self.l58(x5)))
        x8 = torch.cat([x8,a],2)
        x6 = F.relu(self.bn(self.l6(x8) + self.l36(x3)))
        x6 = torch.cat([x6,a],2)
        x4 = torch.tanh(self.bn(self.l4(x6) + self.l14(x1)))
        x4 = torch.cat([x4,a],2)
        ns = self.l2(x4) + self.l02(s)
        r = torch.cat([ns,x7],2)

        # x1 = F.relu(self.bn(self.l1(s)))
        # x3 = F.relu(self.bn(self.l3(x1)))
        # x50 = F.relu(self.bn(self.l5(x3)))
        # x5 = torch.cat([x50,z,a],2)
        # x6 = F.relu(self.bn(self.l6(x5) + self.l36(x3)))
        # x6 = torch.cat([x6,a],2)
        # x4 = torch.tanh(self.bn(self.l4(x6) + self.l14(x1)))
        # x4 = torch.cat([x4,a],2)
        # ns = self.l2(x4) + self.l02(s)
        # r = torch.cat([ns,x50],2)
        
        # x1 = F.relu(self.bn(self.l1(s)))
        # x30 = F.relu(self.bn(self.l3(x1)))
        # x3 = torch.cat([x30,z,a],2)
        # x4 = F.relu(self.bn(self.l4(x3) + self.l14(x1)))
        # x4 = torch.cat([x4,a],2)
        # ns = self.l2(x4) + self.l02(s)
        # r = torch.cat([ns,x30],2)   
        
        # x1 = F.relu(self.bn(self.l1(s)))
        # x3 = F.relu(self.bn(self.l3(x1)))
        # x50 = F.relu(self.bn(self.l5(x3)))
        # x5 = torch.cat([x50,z,a],2)
        # x6 = F.relu(self.bn(self.l6(x5)))
        # x6 = torch.cat([x6,a],2)
        # x4 = torch.tanh(self.bn(self.l4(x6)))
        # x4 = torch.cat([x4,a],2)
        # ns = self.l2(x4)
        # r = torch.cat([ns,x50],2)     

        # x = torch.cat([s.unsqueeze(1).repeat(1,z.size(1),1),z,a],2)
        # x = F.relu(self.bn(self.l1(x)))
        # x = F.relu(self.bn(self.l2(x)))
        # ns = F.relu(self.bn(self.l3(x)))
        
        r = self.l_reward(r)
        return ns, r

class SimPLeNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, beta=5.0e0, max_C=np.log(2), delta_C=np.log(2)*1.0e-5, reward_weight=1.0, C_0=0.0, distribution_type='discrete', alpha=0.99):
        super().__init__()      
        
        self.encoder = SimPLe_encoderNet(s_dim, a_dim, n_tasks=n_tasks, lr=lr, distribution_type=distribution_type)
        self.decoder = SimPLe_decoderNet(s_dim, a_dim, n_tasks=n_tasks, lr=lr, latent_dim=self.encoder.out_dim)
        self.n_tasks = n_tasks
        self.beta = beta
        self.max_C = max_C
        self.delta_C = delta_C
        self.reward_weight = reward_weight
        self.C = C_0
        self.encoder_distribution_type = distribution_type
        self.alpha = alpha

        self.optimizer = optim.Adam(self.parameters(), lr=lr) 

        self.min_values = 1e2*torch.ones(1,s_dim).to(device)
        self.max_values = -1e2*torch.ones(1,s_dim).to(device) 
        self.estimated_mean = torch.zeros(1,s_dim).to(device)
        self.estimated_std = torch.zeros(1,s_dim).to(device)
        self.reward_mean = 0.0
        self.reward_std = 0.0

        # self.mean_stdev = 1.0  
    
    def forward(self, s, a, t, ns=[]):
        training = len(ns) > 0
        if training:
            # self.min_values = (torch.min(torch.stack([ns.min(0,keepdim=True)[0], self.min_values]),0)[0])
            # self.max_values = (torch.max(torch.stack([ns.max(0,keepdim=True)[0], self.max_values]),0)[0])
            self.estimated_mean = self.alpha * self.estimated_mean + (1.0-self.alpha) * 0.5 * (s.mean(0, keepdim=True) + ns.mean(0, keepdim=True)).detach()
            self.estimated_std = self.alpha * self.estimated_std + (1.0-self.alpha) * 0.5 * (((s-self.estimated_mean)**2).mean(0, keepdim=True)**0.5 + ((ns-self.estimated_mean)**2).mean(0, keepdim=True)**0.5).detach()

            z, mean, log_stdev = self.encoder(s,a,ns)

            # self.mean_stdev += 0.1 *(torch.exp(log_stdev).mean().item()-self.mean_stdev)
        else:
            if self.encoder_distribution_type == 'discrete':
                r = torch.rand(s.size(0),self.n_tasks,self.encoder.out_dim).to(device)
                z = (r > 0.5).float()
            else:
                z = torch.randn(s.size(0),self.n_tasks,self.encoder.out_dim).to(device) # * self.mean_stdev

        ns, r = self.decoder(s,a.unsqueeze(1).repeat(1,self.n_tasks,1),z,self.min_values,self.max_values)
        
        if training:
            return ns, r, mean, log_stdev, z
        else:
            return ns, r
    
    def loss_func(self, ns_off, ns, r_off, r, mean, log_stdev, z):
        self.reward_mean = self.alpha * self.reward_mean + (1.0-self.alpha) * r.mean().detach()
        self.reward_std = self.alpha * self.reward_std + (1.0-self.alpha) * (((r-self.reward_mean)**2).mean()**0.5).detach()


        if self.encoder_distribution_type == 'discrete':
            posterior_error = (z * torch.log(mean + 1e-10) + (1-z) * torch.log((1.0-mean).clamp(1e-10,1.0)) + np.log(2)).sum(1).mean()
        else:
            posterior_error = (0.5*(mean**2 + torch.exp(log_stdev)**2 - 1) - log_stdev).sum(1).mean()
        # reconstruction_error = (((ns_off - ns) / ((self.max_values-self.min_values).abs()+1e-2))**2).sum(1).mean()
        reconstruction_error = (((ns_off - ns) / (self.estimated_std+1e-6))**2).sum(1).mean()
        reward_error = (((r_off-r) / (self.reward_std+1e-6))**2).mean()
        loss = self.beta * (posterior_error - self.C).abs() + reconstruction_error + self.reward_weight * reward_error #).clamp(0.0,100.0)

        self.C = np.min([self.C + self.delta_C, self.max_C])

        return loss

class ConditionalSimPLe_encoderNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, distribution_type='discrete', min_T=5e-1, T_0=1.0, annealing_steps=1e5):
        super().__init__()    

        self.distribution_type = distribution_type
        self.out_dim = s_dim//4+1
        self.n_tasks = n_tasks
        if distribution_type == 'discrete':
            self.T = T_0
            self.min_T = min_T
            self.decrease_rate = (min_T/T_0)**(1.0/annealing_steps)
            self.annealing_steps = annealing_steps
        
        self.l1 = nn.Linear(2*s_dim+a_dim+n_tasks, s_dim)
        self.l2 = nn.Linear(s_dim, s_dim//2+1)
        self.l31 = nn.Linear(s_dim//2+1, self.out_dim)
        if distribution_type != 'discrete':
            self.l32 = nn.Linear(s_dim//2+1, self.out_dim)

        self.apply(weights_init_)

    def forward(self, s, a, ns, t):
        self.T = np.max([self.decrease_rate * self.T, self.min_T])

        x = torch.cat([s,a,ns,t],1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        if self.distribution_type == 'discrete':            
            mean = torch.sigmoid(self.l31(x))
            mean = mean / mean.sum(1, keepdim=True)
            
            u = torch.rand_like(mean)
            g = -torch.log((-torch.log(u+1e-10)).clamp(1e-10,1.0))
            y = torch.exp((torch.log(mean+1e-10) + g) / self.T)
            y = y / y.sum(1, keepdim=True)
            z = y

            log_stdev = mean
            
        else:
            mean = self.l31(x)
            log_stdev = self.l32(x)
            z = mean + torch.exp(log_stdev) * torch.randn_like(mean) #) * self.max_log_stdev
        
        return z, mean, log_stdev

class ConditionalSimPLe_decoderNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, latent_dim=0):
        super().__init__()      
        dim1 = (5*s_dim)//6+1
        dim2 = (3*s_dim)//4+1
        dim3 = (1*s_dim)//2+1
        dim4 = (1*s_dim)//4+1

        self.l1 = nn.Linear(s_dim, dim1)
        self.l3 = nn.Linear(dim1, dim2)
        self.l5 = nn.Linear(dim2, dim3)
        self.l7 = nn.Linear(dim3, dim4)
        self.l8 = nn.Linear(dim4, dim3)
        self.l6 = nn.Linear(dim3, dim2)
        self.l4 = nn.Linear(dim2, dim1)
        self.l2 = nn.Linear(dim1, s_dim)
        
        # Skip - connection layers
        self.l02 = nn.Linear(s_dim, s_dim)
        self.l14 = nn.Linear(dim1, dim1)
        self.l36 = nn.Linear(dim2, dim2)
        self.l58 = nn.Linear(dim3, dim3)

        self.la8 = nn.Linear(a_dim+latent_dim+n_tasks, dim3)
        self.la6 = nn.Linear(a_dim+latent_dim+n_tasks, dim2)
        self.la4 = nn.Linear(a_dim+latent_dim+n_tasks, dim1)
        self.la2 = nn.Linear(a_dim+latent_dim+n_tasks, s_dim)

        self.l_reward = nn.Linear(s_dim+a_dim+dim4+latent_dim+n_tasks, 1)

        self.bn1 = nn.BatchNorm1d(dim1)
        self.bn3 = nn.BatchNorm1d(dim2)
        self.bn5 = nn.BatchNorm1d(dim3)
        self.bn7 = nn.BatchNorm1d(dim4)
        self.bn8 = nn.BatchNorm1d(dim3)
        self.bn6 = nn.BatchNorm1d(dim2)
        self.bn4 = nn.BatchNorm1d(dim1)
        self.d = nn.Dropout(0.2)
                
        self.apply(weights_init_)
    
    def forward(self, s, a, z, t, min_, max_):
        attention_input = torch.cat([z,a,t],1)
        x1 = F.relu(self.bn1(self.l1(s)))
        x3 = F.relu(self.bn3(self.l3(x1)))
        x5 = F.relu(self.bn5(self.l5(x3)))
        x7 = F.relu(self.bn7(self.l7(x5)))
        # x7 = torch.cat([x_abstract,z,a,t],1)
        x8 = (F.relu(self.bn8(self.l8(x7) + self.l58(x5))))*torch.sigmoid(self.la8(attention_input))#torch.cat([x5,t],1))))
        # x8 = torch.cat([x8,a],1)
        x6 = (F.relu(self.bn6(self.l6(x8) + self.l36(x3))))*torch.sigmoid(self.la6(attention_input))
        # x6 = torch.cat([x6,a],1)
        x4 = (F.relu(self.bn4(self.l4(x6) + self.l14(x1))))*torch.sigmoid(self.la4(attention_input))
        # x4 = torch.cat([x4,a],1)
        ns = (self.l2(x4) + self.l02(s))*torch.sigmoid(self.la2(attention_input))
        r = torch.cat([ns,x7,z,a,t],1)
        
        r = self.l_reward(r)
        return ns, r

class ConditionalSimPLeNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, beta=1.0e2, max_C=10*np.log(2), delta_C=np.log(2)*1.0e-4, reward_weight=1.0, C_0=0.0, distribution_type='discrete', alpha=0.99):
        super().__init__()      
        
        self.encoder = ConditionalSimPLe_encoderNet(s_dim, a_dim, n_tasks=n_tasks, lr=lr, distribution_type=distribution_type)
        self.latent_dim = self.encoder.out_dim
        self.decoder = ConditionalSimPLe_decoderNet(s_dim, a_dim, n_tasks=n_tasks, lr=lr, latent_dim=self.latent_dim)
        self.n_tasks = n_tasks
        self.beta = beta
        self.max_C = max_C
        self.delta_C = delta_C
        self.reward_weight = reward_weight
        self.C = C_0
        self.encoder_distribution_type = distribution_type
        self.alpha = alpha

        self.optimizer = optim.Adam(self.parameters(), lr=lr) 

        self.min_values = 1e2*torch.ones(1,s_dim).to(device)
        self.max_values = -1e2*torch.ones(1,s_dim).to(device) 
        self.estimated_mean = torch.zeros(1,s_dim).to(device)
        self.estimated_std = torch.zeros(1,s_dim).to(device)
        self.reward_mean = 0.0
        self.reward_std = 0.0

        # self.mean_stdev = 1.0  
    
    def forward(self, s, a, t, ns=[], z=[]):
        t_mask = torch.zeros(s.size(0), self.n_tasks).to(device)
        t_mask[np.arange(s.size(0)), t] = torch.ones(s.size(0),).to(device)

        training = len(ns) > 0
        if training:
            # self.min_values = (torch.min(torch.stack([ns.min(0,keepdim=True)[0], self.min_values]),0)[0])
            # self.max_values = (torch.max(torch.stack([ns.max(0,keepdim=True)[0], self.max_values]),0)[0])
            self.estimated_mean = self.alpha * self.estimated_mean + (1.0-self.alpha) * 0.5 * (s.mean(0, keepdim=True) + ns.mean(0, keepdim=True)).detach()
            self.estimated_std = self.alpha * self.estimated_std + (1.0-self.alpha) * 0.5 * (((s-self.estimated_mean)**2).mean(0, keepdim=True)**0.5 + ((ns-self.estimated_mean)**2).mean(0, keepdim=True)**0.5).detach()

            z, mean, log_stdev = self.encoder(s,a,ns,t_mask)

            # self.mean_stdev += 0.1 *(torch.exp(log_stdev).mean().item()-self.mean_stdev)
        else:
            if len(z) <= 0:
                if self.encoder_distribution_type == 'discrete':
                    r = torch.rand(s.size(0),self.encoder.out_dim).to(device)
                    z = (r > 0.5).float()
                else:
                    z = torch.randn(s.size(0),self.encoder.out_dim).to(device) # * self.mean_stdev

        ns, r = self.decoder(s,a,z,t_mask,self.min_values,self.max_values)
        
        if training:
            return ns, r, mean, log_stdev, z
        else:
            return ns, r
    
    def loss_func(self, ns_off, ns, r_off, r, mean, log_stdev, z):
        self.reward_mean = self.alpha * self.reward_mean + (1.0-self.alpha) * r.mean().detach()
        self.reward_std = self.alpha * self.reward_std + (1.0-self.alpha) * (((r-self.reward_mean)**2).mean()**0.5).detach()

        if self.encoder_distribution_type == 'discrete':
            posterior_error = (mean * torch.log(mean + 1e-10) + (1-mean) * torch.log((1.0-mean).clamp(1e-10,1.0)) + np.log(2)).sum(1).mean()
        else:
            posterior_error = (0.5*(mean**2 + torch.exp(log_stdev)**2 - 1) - log_stdev).sum(1).mean()
        reconstruction_error = (((ns_off - ns) / (self.estimated_std+1e-6))**2).sum(1).mean()
        reward_error = (((r_off-r) / (self.reward_std+1e-6))**2).mean()
        loss = self.beta * (posterior_error - self.C).abs() + reconstruction_error + self.reward_weight * reward_error #).clamp(0.0,100.0)

        self.C = np.min([self.C + self.delta_C, self.max_C])

        print("posterior loss: "+ str(np.round(self.beta * (posterior_error - self.C).abs().item(),4)))
        print("reconstruction loss: "+ str(np.round(reconstruction_error.item(),4)))
        print("reward reconstruction loss: "+ str(np.round(self.reward_weight * reward_error.item(),4)))

        return loss

class AutoregressivePrior(nn.Module):
    def __init__(self, z_dim, period=1, n_tasks=1, lr=3e-4, wd=1e-2):
        super().__init__()

        self.length = z_dim*period

        self.l1 = AutoregressiveLinear(self.length)
        self.l2 = AutoregressiveLinear(self.length)
        self.l3 = AutoregressiveLinear(self.length)

        self.a1 = nn.Linear(n_tasks, self.length)
        self.a2 = nn.Linear(n_tasks, self.length)
        self.a3 = nn.Linear(n_tasks, self.length)

        self.apply(weights_init_) 
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)
    
    def forward(self, z, t):
        x = F.relu(self.l1(z)) * torch.sigmoid(self.a1(t))
        x = F.relu(self.l2(x)) * torch.sigmoid(self.a2(t))
        p = torch.sigmoid(self.l3(x)) * torch.sigmoid(self.a3(t))
        return p

    def sample(self, t):
        z = torch.ones(t.size(0), self.length).to(device)
        for i in range(0,self.length):
            p = self(z,t)
            z[:,i] = (torch.rand(t.size(0)).to(device) < p[:,i])*1.0
        return z


class ConditionalVQVAE_encoderNet(nn.Module):
    def __init__(self, s_dim, vision_dim=20, vision_channels=2):
        super().__init__()
        self.vision_channels = vision_channels
        self.vision_dim = vision_dim
        self.kinematic_dim = s_dim - vision_dim*vision_channels     
        
        nc1 = vision_channels * 2
        nc2 = vision_channels * 3
        nc3 = vision_channels * 4

        kernel_size = 2
        stride1 = 2
        stride2 = 2
        stride3 = 1
        
        k_dim1 = (2*self.kinematic_dim)//3+1
        k_dim2 = (3*k_dim1)//5
        k_dim3 = (2*k_dim2)//3

        v_dim1 = int((vision_dim - kernel_size)/stride1 + 1)
        v_dim2 = int((v_dim1 - kernel_size)/stride2 + 1)
        v_dim3 = int((v_dim2 - kernel_size)/stride3 + 1)
        
        self.out_channels = nc3
        self.out_dim = v_dim3+k_dim3

        self.lv1 = nn.Conv1d(vision_channels, nc1, kernel_size, stride=stride1)
        self.lv2 = nn.Conv1d(nc1, nc2, kernel_size, stride=stride2)
        self.lv3 = nn.Conv1d(nc2, nc3, kernel_size, stride=stride3)

        self.lk1 = multichannel_Linear(1, nc1, self.kinematic_dim, k_dim1)
        self.lk2 = multichannel_Linear(nc1, nc2, k_dim1, k_dim2)
        self.lk3 = multichannel_Linear(nc2, nc3, k_dim2, k_dim3)

        self.lm1 = parallel_Linear(nc3, self.out_dim, self.out_dim)
        self.lm2 = parallel_Linear(nc3, self.out_dim, self.out_dim)
        
        self.bn1 = nn.BatchNorm1d(nc1)
        self.bn2 = nn.BatchNorm1d(nc2)
        self.bn3 = nn.BatchNorm1d(nc3)
                        
        self.apply(weights_init_)
    
    def forward(self, s):
        vision_input = s[:,-int(self.vision_dim*self.vision_channels):].view(s.size(0),self.vision_channels,self.vision_dim)
        kinematic_input = s[:,:-int(self.vision_dim*self.vision_channels)].unsqueeze(1)

        v = F.relu(self.bn1(self.lv1(vision_input)))
        assert v.size(2) == 10, 'wtf vdim'
        v = F.relu(self.bn2(self.lv2(v)))
        assert v.size(2) == 5, 'wtf vdim'
        v = self.lv3(v)
        assert v.size(2) == 4, 'wtf vdim'

        k = F.relu(self.bn1(self.lk1(kinematic_input)))
        k = F.relu(self.bn2(self.lk2(k)))
        k = self.lk3(k)

        ze = torch.cat([k,v],2)
        ze = F.relu(self.bn3(self.lm1(ze)))
        ze = self.lm2(ze)

        return ze

class ConditionalVQVAE_decoderNet(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, vision_dim=20, vision_channels=2):
        super().__init__()      
        self.n_tasks = n_tasks
        self.vision_channels = vision_channels
        self.vision_dim = vision_dim
        self.kinematic_dim = s_dim - vision_dim*vision_channels      
        
        nc1 = vision_channels * 2
        nc2 = vision_channels * 3
        nc3 = vision_channels * 4

        kernel_size = 2
        stride1 = 2
        stride2 = 2
        stride3 = 1
        
        k_dim1 = (2*self.kinematic_dim)//3+1
        k_dim2 = (3*k_dim1)//5
        k_dim3 = (2*k_dim2)//3

        v_dim1 = int((vision_dim - kernel_size)/stride1 + 1)
        v_dim2 = int((v_dim1 - kernel_size)/stride2 + 1)
        v_dim3 = int((v_dim2 - kernel_size)/stride3 + 1)
        
        self.in_channels = nc3
        self.in_dim = v_dim3+k_dim3
        self.latent_vision_dim = v_dim3

        self.lv1 = nn.ConvTranspose1d(nc3, nc2, kernel_size, stride=stride3)
        self.lv2 = nn.ConvTranspose1d(nc2, nc1, kernel_size, stride=stride2)
        self.lv3 = nn.ConvTranspose1d(nc1, vision_channels, kernel_size, stride=stride1)
        
        self.lk1 = multichannel_Linear(nc3, nc2, k_dim3, k_dim2)
        self.lk2 = multichannel_Linear(nc2, nc1, k_dim2, k_dim1)
        self.lk3 = multichannel_Linear(nc1, 1, k_dim1, self.kinematic_dim)
        
        self.la1 = nn.Linear(a_dim+n_tasks, self.in_dim*nc3)
        self.la2 = nn.Linear(a_dim+n_tasks, self.in_dim*nc3)
        self.lak1 = nn.Linear(a_dim+n_tasks, k_dim2*nc2)
        self.lak2 = nn.Linear(a_dim+n_tasks, k_dim1*nc1)
        self.lak3 = nn.Linear(a_dim+n_tasks, self.kinematic_dim)
        self.lav1 = nn.Linear(a_dim+n_tasks, v_dim2*nc2)
        self.lav2 = nn.Linear(a_dim+n_tasks, v_dim1*nc1)
        self.lav3 = nn.Linear(a_dim+n_tasks, self.vision_dim*self.vision_channels)
        self.lar1 = nn.Linear(n_tasks, 100)

        self.lm1 = parallel_Linear(nc3, self.in_dim, self.in_dim)
        self.lm2 = parallel_Linear(nc3, self.in_dim, self.in_dim)

        self.lr1 = nn.Linear(s_dim+a_dim+self.in_dim*nc3, 100)
        self.lr2 = nn.Linear(100, 1)
        
        self.bn1 = nn.BatchNorm1d(nc3)
        self.bn2 = nn.BatchNorm1d(nc2)
        self.bn3 = nn.BatchNorm1d(nc1)
        self.bnr = nn.BatchNorm1d(100)
                
        self.apply(weights_init_)
    
    def forward(self, zq, a, t):
        t_mask = torch.zeros(a.size(0), self.n_tasks).to(device)
        t_mask[np.arange(a.size(0)), t] = torch.ones(a.size(0),).to(device)
        attention_input = torch.cat([a,t_mask],1)

        x = F.relu(self.bn1(self.lm1(zq))) * torch.sigmoid(self.la1(attention_input)).view(-1,self.in_channels, self.in_dim)
        x = F.relu(self.bn1(self.lm2(x))) * torch.sigmoid(self.la2(attention_input)).view(-1,self.in_channels, self.in_dim)

        v = x[:,:,-self.latent_vision_dim:]
        k = x[:,:,:-self.latent_vision_dim]

        k = F.relu(self.bn2(self.lk1(k)))
        k = k * torch.sigmoid(self.lak1(attention_input)).view(k.size())
        k = F.relu(self.bn3(self.lk2(k)))
        k = k * torch.sigmoid(self.lak2(attention_input)).view(k.size())
        k = self.lk3(k)
        k = k.squeeze(1) * torch.sigmoid(self.lak3(attention_input))

        v = F.relu(self.bn2(self.lv1(v)))
        v = v * torch.sigmoid(self.lav1(attention_input)).view(v.size())
        v = F.relu(self.bn3(self.lv2(v)))
        v = v * torch.sigmoid(self.lav2(attention_input)).view(v.size())
        v = self.lv3(v)
        v = v.view(-1,self.vision_dim*self.vision_channels) * torch.sigmoid(self.lav3(attention_input))

        ns = torch.cat([k,v],1)
        
        r = torch.cat([ns,a,zq.view(-1,self.in_dim*self.in_channels)],1)
        r = F.relu(self.bnr(self.lr1(r))) * torch.sigmoid(self.lar1(t_mask))
        r = self.lr2(r)
        
        return ns, r

class ConditionalVQVAE_embeddingSpaceNet(nn.Module):
    def __init__(self, s_dim, n_embedding_vectors=512, std_init=1.0, embedding_dim=1):
        super().__init__()      
        self.embedding_dim = embedding_dim

        self.n_embedding_vectors = n_embedding_vectors
        self.dictionary = Parameter(torch.Tensor(n_embedding_vectors, embedding_dim))
        nn.init.normal_(self.dictionary, mean=0.0, std=std_init)
    
    def code2latent(self, ze):
        z = ((self.dictionary.unsqueeze(0).unsqueeze(2) - ze.unsqueeze(1))**2).sum(3).argmin(1)
        return z
    
    def forward(self, ze, z=[]):
        if len(z) == 0:
            z = self.code2latent(ze)
        e = self.dictionary[z.view(-1),:].view(z.size(0), z.size(1), -1)
        zq = e.clone()
        return zq, e

class ConditionalVQVAE_Net(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks=1, lr=3e-4, beta=10.0, reward_weight=1.0, alpha=0.99):
        super().__init__()

        self.encoder = ConditionalVQVAE_encoderNet(s_dim)
        self.decoder = ConditionalVQVAE_decoderNet(s_dim, a_dim, n_tasks=n_tasks)
        self.embedding_space = ConditionalVQVAE_embeddingSpaceNet(s_dim, embedding_dim=self.encoder.out_dim)
        self.latent_dim = int(np.log2(self.embedding_space.n_embedding_vectors)*self.encoder.out_channels)
        self.n_channels = self.encoder.out_channels

        self.alpha = alpha
        self.beta = beta
        self.reward_weight = reward_weight

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.estimated_mean = torch.zeros(1,s_dim).to(device)
        self.estimated_std = torch.zeros(1,s_dim).to(device)
        self.reward_mean = 0.0
        self.reward_std = 0.0
    
    def forward(self, s, a, t, z=[]):
        self.estimated_mean = self.alpha * self.estimated_mean + (1.0-self.alpha) * s.mean(0, keepdim=True).detach()
        self.estimated_std = self.alpha * self.estimated_std + (1.0-self.alpha) * (((s-self.estimated_mean)**2).mean(0, keepdim=True)**0.5).detach()

        ze = self.encoder(s)
        zq, e = self.embedding_space(ze, z=z)
        zq = zq.detach() + ze - ze.detach()
        ns, r = self.decoder(zq, a, t)
        return ns, r, ze, e
    
    def loss_func(self, ns_off, ns, r_off, r, ze, e):
        self.reward_mean = self.alpha * self.reward_mean + (1.0-self.alpha) * r.mean().detach()
        self.reward_std = self.alpha * self.reward_std + (1.0-self.alpha) * (((r-self.reward_mean)**2).mean()**0.5).detach()

        reconstruction_error = (((ns_off - ns) / (self.estimated_std+1e-6))**2).sum(1).mean()
        reward_error = (((r_off-r) / (self.reward_std+1e-6))**2).mean()
        VQ_loss = ((ze.detach()-e)**2).sum(2).mean()
        commitment_loss = ((ze-e.detach())**2).sum(2).mean()

        loss = reconstruction_error + self.reward_weight * reward_error + VQ_loss + self.beta * commitment_loss

        print("reconstruction loss: "+ str(np.round(reconstruction_error.item(),4)))
        print("reward reconstruction loss: "+ str(np.round(self.reward_weight * reward_error.item(),4)))
        print("VQ loss: "+ str(np.round(VQ_loss.item(),4)))
        print("commitment loss: "+ str(np.round(commitment_loss.item(),4)))

        return loss


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

class AutoregressiveLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = Parameter(torch.Tensor(dim, dim))
        self.bias = Parameter(torch.Tensor(dim))
        self.mask = torch.zeros(dim, dim).to(device)        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        x,y = np.tril_indices(self.dim)
        self.mask[x,y] = torch.ones((self.dim*(self.dim+1))//2).to(device)

    def forward(self, input):
        return torch.einsum('ik,lk->il', input, self.weight*self.mask) + self.bias.unsqueeze(0) 

    def extra_repr(self):
        return 'dim={}, bias={}'.format(
            self.dim, self.bias is not None
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

class multichannel_Linear(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.weight = Parameter(torch.Tensor(n_channels_in, n_channels_out, in_features, out_features))
        self.bias = Parameter(torch.Tensor(n_channels_out, out_features))        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.einsum('ijk,jlkm->ilm', input, self.weight) + self.bias.unsqueeze(0) 

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

class r_parallelNet(nn.Module):
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
        x = self.l3(x)
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
    def __init__(self, n_m_states, input_dim, n_tasks=1, hidden_dim=256, min_log_stdev=-4, max_log_stdev=2, lr=3e-4, min_c=1, init_method='glorot'):
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
        s_repeated = s.clone().view(s.size(0),1,s.size(1)).repeat(1,self.n_m_states,1)
        x = F.relu(self.l1(s_repeated))
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
            llhoods = gaussian_likelihood(s_repeated, m, log_stdev, self.EPS_sigma)
        elif self.log_func == 'torch':
            llhoods = Normal(m, torch.exp(log_stdev)).log_prob(s_repeated)
        assert torch.all(llhoods==llhoods), 'Invalid memb llhoods'

        llhoods = torch.clamp(llhoods, self.min_log_stdev, self.max_log_stdev).sum(2)
        
        return llhoods, m, log_stdev

    def sample_m_state(self, s, explore=True):
        llhoods = self(s.view(1,-1))[0].view(-1) + torch.log(self.prior.mean(0)/self.prior_n+1e-12)
        lmmbrship = torch.logsumexp(llhoods,0)
        log_posterior = (llhoods-lmmbrship).clamp(self.min_log_stdev*np.log(self.n_m_states),0.0)
        log_posterior = log_posterior - torch.logsumexp(log_posterior,0)
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
        llhoods = self(s)[0] + torch.log(self.prior.mean(0)/self.prior_n+1e-12).view(1,-1)
        lmmbrship = torch.logsumexp(llhoods, 1, keepdim=True)        
        log_posterior = (llhoods-lmmbrship).clamp(self.min_log_stdev*np.log(self.n_m_states),0.0)
        log_posterior = log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True)
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
        x = s.clone().view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        if self.latent_dim > 0:
            t = torch.randn(x.size(0), 1, self.latent_dim).repeat(1,self.n_m_actions,1).float().cuda()
            x = torch.cat([x,t], 2)
        x = F.relu(self.l11(x))
        x = F.relu(self.l21(x))
        m = self.l31(x).clone()
        return m
    
    def sample_action_and_llhood_pairs(self, s, A, explore=True):
        x = s.clone().view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(x)
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
        xs = s.clone().view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        xa = a.clone().view(a.size(0),1,a.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(xs)
        stdev = log_stdev.exp()
        u = atanh(xa)

        if self.log_func == 'self':
            llhoods = gaussian_likelihood(u, m, log_stdev, self.EPS_sigma)
        elif self.log_func == 'torch':
            llhoods = Normal(m, stdev).log_prob(u)

        if self.log_lim_method == 'clamp':
            llhoods -= torch.log(torch.clamp(1 - xa.pow(2), self.EPS_log_1_min_a2, 1.0))    
        elif self.log_lim_method == 'sum':
            llhoods -= torch.log(1 - xa.pow(2) + self.EPS_log_1_min_a2)

        llhoods = llhoods.sum(2)       

        return llhoods
    
    def sample_actions(self, s, repeat=True):
        x = s.clone()
        if repeat:
            x = x.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(x)
        stdev = log_stdev.exp()
        u = m + stdev*torch.randn_like(m)
        a = torch.tanh(u)
        return a
    
    def sample_actions_and_llhoods_for_all_skills(self, s, explore=True):
        x = s.clone().view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1)
        m, log_stdev = self(x)
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
            x = torch.cat([x,t], 2)
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

