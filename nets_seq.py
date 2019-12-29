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
    return likelihood

def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

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

    @property
    def empty(self):
        return len(self.data) == 0
    
    @property
    def len_data(self):
        return len(self.data)

#-------------------------------------------------------------
#
#    Value networks
#
#-------------------------------------------------------------
class v_Net(nn.Module):
    def __init__(self, input_dim, n_tasks, lr=3e-4):
        super().__init__()        
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_tasks)  

        self.apply(weights_init_)       
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)           
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

class q_Net(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks, lr=3e-4):
        super().__init__()        
        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_tasks)  

        self.apply(weights_init_) 
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

class m_Net(nn.Module):
    def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
        super().__init__()        
        self.l1 = nn.Linear(n_concepts+n_tasks, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, n_skills)  

        self.apply(weights_init_) 
        
    def forward(self, S,T):
        x = torch.cat([S, T], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.exp(self.l3(x))
        PA_ST = x / x.sum(1, keepdim=True)
        return(PA_ST)

class DQN(nn.Module):
    def __init__(self, s_dim, n_skills, n_tasks, vision_dim=20, vision_channels=3, lr=3e-4):
        super().__init__()  
        self.s_dim = s_dim
        self.n_skills = n_skills
        self.n_tasks = n_tasks   
        self.vision_dim = vision_dim
        self.vision_channels = vision_channels
        self.kinematic_dim = s_dim - vision_dim*vision_channels     
        
        nc1 = vision_channels * 2
        nc2 = vision_channels * 4
        nc3 = vision_channels * 8
        nc4 = vision_channels * 16

        kernel_size1 = 4
        kernel_size2 = 4
        kernel_size3 = 3
        kernel_size4 = 3

        dilation1 = 1
        dilation2 = 2
        dilation3 = 1
        dilation4 = 2
        
        k_dim1 = 24
        k_dim2 = 20
        k_dim3 = 15
        k_dim4 = 10

        v_dim1 = int((vision_dim - dilation1*(kernel_size1-1) - 1)/1 + 1)
        v_dim2 = int((v_dim1 - dilation2*(kernel_size2-1) - 1)/1 + 1)
        v_dim3 = int((v_dim2 - dilation3*(kernel_size3-1) - 1)/1 + 1)
        v_dim4 = int((v_dim3 - dilation4*(kernel_size4-1) - 1)/1 + 1)
        
        self.lv1e = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
        self.lv2e = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
        self.lv3e = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
        self.lv4e = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)
        self.lv1g = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
        self.lv2g = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
        self.lv3g = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
        self.lv4g = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)        

        self.lk1 = multichannel_Linear(1, nc1, self.kinematic_dim, k_dim1)
        self.lk2 = multichannel_Linear(nc1, nc2, k_dim1, k_dim2)
        self.lk3 = multichannel_Linear(nc2, nc3, k_dim2, k_dim3)
        self.lk4 = multichannel_Linear(nc3, nc4, k_dim3, k_dim4)

        self.lc1x1 = nn.Conv1d(nc4, n_tasks, 1, stride=1)
        self.lkv1 = parallel_Linear(n_tasks, v_dim4+k_dim4, 256)
        self.lkv2 = parallel_Linear(n_tasks, 256, n_skills)
        
        self.bn1 = nn.BatchNorm1d(nc1)
        self.bn2 = nn.BatchNorm1d(nc2)
        self.bn3 = nn.BatchNorm1d(nc3)
        self.bn4 = nn.BatchNorm1d(nc4)
        self.bn5 = nn.BatchNorm1d(n_tasks)
                        
        self.apply(weights_init_)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)       

    def forward(self, s):
        vision_input = s[:,-int(self.vision_dim*self.vision_channels):].view(s.size(0),self.vision_channels,self.vision_dim)
        kinematic_input = s[:,:-int(self.vision_dim*self.vision_channels)].unsqueeze(1)

        v = torch.tanh(self.bn1(self.lv1e(vision_input))) * torch.sigmoid(self.bn1(self.lv1g(vision_input)))
        v = torch.tanh(self.bn2(self.lv2e(v))) * torch.sigmoid(self.bn2(self.lv2g(v)))
        v = torch.tanh(self.bn3(self.lv3e(v))) * torch.sigmoid(self.bn3(self.lv3g(v)))
        v = torch.tanh(self.bn4(self.lv4e(v))) * torch.sigmoid(self.bn4(self.lv4g(v)))
        
        k = F.relu(self.bn1(self.lk1(kinematic_input)))
        k = F.relu(self.bn2(self.lk2(k)))
        k = F.relu(self.bn3(self.lk3(k)))
        k = torch.tanh(self.bn4(self.lk4(k)))

        x = torch.cat([k,v],2)
        x = F.relu(self.bn5(self.lc1x1(x)))
        x = F.relu(self.bn5(self.lkv1(x)))
        x = self.lkv2(x)

        return x    

    #     self.l1 = nn.Linear(s_dim, 256)
    #     self.l2 = nn.Linear(256, 256)
    #     self.l3 = nn.Linear(256, n_tasks*n_skills)  

    #     self.apply(weights_init_) 
    #     self.loss_func = nn.MSELoss()
    #     self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    # def forward(self, s):
    #     x = F.relu(self.l1(s))
    #     x = F.relu(self.l2(x))
    #     x = self.l3(x).view(-1, self.n_tasks, self.n_skills)
    #     return(x)

#-------------------------------------------------------------
#
#    Custom modules
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

#-------------------------------------------------------------
#
#    Concept and skill nets
#
#-------------------------------------------------------------
class c_Net(nn.Module):
    def __init__(self, n_concepts, s_dim, n_skills, n_tasks=1, lr=3e-4, vision_dim=20, vision_channels=2):
        super().__init__()  
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_concepts = n_concepts
        self.vision_channels = vision_channels
        self.vision_dim = vision_dim
        self.kinematic_dim = s_dim - vision_dim*vision_channels     
        
        nc1 = vision_channels * 2
        nc2 = vision_channels * 4
        nc3 = vision_channels * 8
        nc4 = vision_channels * 16

        kernel_size1 = 4
        kernel_size2 = 4
        kernel_size3 = 3
        kernel_size4 = 3

        dilation1 = 1
        dilation2 = 2
        dilation3 = 1
        dilation4 = 2
        
        k_dim1 = 24
        k_dim2 = 20
        k_dim3 = 15
        k_dim4 = 10

        v_dim1 = int((vision_dim - dilation1*(kernel_size1-1) - 1)/1 + 1)
        v_dim2 = int((v_dim1 - dilation2*(kernel_size2-1) - 1)/1 + 1)
        v_dim3 = int((v_dim2 - dilation3*(kernel_size3-1) - 1)/1 + 1)
        v_dim4 = int((v_dim3 - dilation4*(kernel_size4-1) - 1)/1 + 1)
        
        self.lv1e = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
        self.lv2e = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
        self.lv3e = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
        self.lv4e = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)
        self.lv1g = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
        self.lv2g = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
        self.lv3g = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
        self.lv4g = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)        

        self.lk1 = multichannel_Linear(1, nc1, self.kinematic_dim, k_dim1)
        self.lk2 = multichannel_Linear(nc1, nc2, k_dim1, k_dim2)
        self.lk3 = multichannel_Linear(nc2, nc3, k_dim2, k_dim3)
        self.lk4 = multichannel_Linear(nc3, nc4, k_dim3, k_dim4)

        self.lc1x1 = nn.Conv1d(nc4, n_concepts, 1, stride=1)
        self.lkv = parallel_Linear(n_concepts, v_dim4+k_dim4, 1)
        
        self.bn1 = nn.BatchNorm1d(nc1)
        self.bn2 = nn.BatchNorm1d(nc2)
        self.bn3 = nn.BatchNorm1d(nc3)
        self.bn4 = nn.BatchNorm1d(nc4)
        self.bn5 = nn.BatchNorm1d(n_concepts)

        self.map = m_Net(n_concepts, n_skills, n_tasks)
                        
        self.apply(weights_init_)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)        

    def forward(self, s):
        vision_input = s[:,-int(self.vision_dim*self.vision_channels):].view(s.size(0),self.vision_channels,self.vision_dim)
        kinematic_input = s[:,:-int(self.vision_dim*self.vision_channels)].unsqueeze(1)

        v = torch.tanh(self.bn1(self.lv1e(vision_input))) * torch.sigmoid(self.bn1(self.lv1g(vision_input)))
        v = torch.tanh(self.bn2(self.lv2e(v))) * torch.sigmoid(self.bn2(self.lv2g(v)))
        v = torch.tanh(self.bn3(self.lv3e(v))) * torch.sigmoid(self.bn3(self.lv3g(v)))
        v = torch.tanh(self.bn4(self.lv4e(v))) * torch.sigmoid(self.bn4(self.lv4g(v)))
        
        k = F.relu(self.bn1(self.lk1(kinematic_input)))
        k = F.relu(self.bn2(self.lk2(k)))
        k = F.relu(self.bn3(self.lk3(k)))
        k = torch.tanh(self.bn4(self.lk4(k)))

        x = torch.cat([k,v],2)
        x = F.relu(self.bn5(self.lc1x1(x)))
        x = torch.exp(self.lkv(x).squeeze(2))
        PS_s = x / x.sum(1, keepdim=True)

        return PS_s
    
    def sample_concept(self, s, explore=True):
        PS_s = self(s.view(1,-1)).view(-1)
        if explore:
            S = Categorical(probs=PS_s).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(PS_s, PS_s.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().item()            
        return S, PS_s

    def sample_skill(self, s, task, explore=True):
        S, PS = self.sample_concept(s, explore=explore)
        PA_ST = self.map(S, task)
        if explore:
            A = Categorical(probs=PA_ST).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().item()
        return A, PA_ST, S, PS
    
    def sample_concepts(self, s, explore=True):
        PS_s = self(s)
        if explore:
            S = Categorical(probs=PS_s).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PS_s, PS_s.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().cpu()            
        return S, PS_s, torch.log(PS_s+1e-12)  

    def sample_skills(self, s, task, explore=True):
        S, PS_s, log_PS_s = self.sample_concepts(s, explore=explore)
        PA_ST = self.map(S, task)
        if explore:
            A = Categorical(probs=PA_ST).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().cpu()
        return A, PA_ST, torch.log(PA_ST+1e-12), S, PS_s, log_PS_s
    
class s_Net(nn.Module):
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
        self.l21 = parallel_Linear(n_m_actions, hidden_dim, hidden_dim)
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
    
    def conditional(self, s, A):
        x = s.clone().view(1,s.size(0))
        if self.latent_dim > 0:
            t = torch.randn(1, self.latent_dim).float().to(device)
            x = torch.cat([x,t], 1)
        x = F.relu(self.l11.conditional(x, A))
        x = F.relu(self.l21.conditional(x, A))
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
        x = s.clone()
        if self.latent_dim > 0:
            t = torch.randn(s.size(0), 1, self.latent_dim).repeat(1,self.n_m_actions,1).float().cuda()
            x = torch.cat([x,t], 2)
        x1 = F.relu(self.l11(x))
        x1 = F.relu(self.l21(x1))
        m = self.l31(x1)
        log_stdev = self.l32(x1)
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

        llhoods = llhoods.sum(3)   

        return a, llhoods


