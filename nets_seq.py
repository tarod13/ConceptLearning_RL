import math
import random
import copy
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

def weights_init_noisy(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(m.bias, a=np.sqrt(5))

def weights_init_big(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.xavier_uniform_(m.weight, gain=10.0)
        torch.nn.init.xavier_uniform_(m.bias, gain=10.0)

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

class Memory_PER:
    def __init__(self, capacity = 50000, n_seed=0):
        self.capacity = capacity
        self.data = []        
        self.pointer = 0
        self.weights = []
        set_seed(n_seed)
    
    def store(self, event):
        if len(self.data) < self.capacity:
            self.data.append(None)
            self.weights.append(None)
        self.data[self.pointer] = event
        self.weights[self.pointer] = 1.0
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample(self, batch_size):
        k = int(batch_size) if batch_size < len(self.data) else len(self.data)
        population = list(np.arange(self.len_data))
        sample = random.choices(population, weights=self.weights, k=k)
        return [self.data[i] for i in sample], sample, torch.FloatTensor([self.weights[i] for i in sample]).to(device)

    def update_weights(self, sample, weights):
        for s, w in zip(sample, weights):
            self.weights[s] = w      

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

class IndexedNode:
    def __init__(self, value, index, root=False, left=None, right=None, code=[]):
        self.root = root        
        self.left = left
        self.right = right
        self.code = code

        if not (left is None or right is None):
            self.value = left.value + right.value
            self.max = right.value # TODO: reorder in case this is not true
            self.min = left.value
            self.left.code = code+[0]
            self.right.code = code+[1]
        elif right is None:
            self.value = left.value
            self.max = left.value
            self.min = left.value
            self.left.code = code+[0]
        else:
            self.value = value
            self.max = value
            self.min = value

        if root:
            self.index = -1            
        else:
            self.index = index

    @property
    def leaf(self):
        return not self.root and self.free
    
    @property 
    def internal_node(self):
        return not (self.leaf or self.root)
    
    @property
    def full(self):
        return not (self.left is None or self.right is None)
    
    @property
    def free(self):
        return self.left is None and self.right is None

class SumTree:
    def __init__(self, alpha=0.0):
        self.root = IndexedNode(0.0, -1, root=True)
    
    def add_node(self, value, index):
        node = IndexedNode(value, index)
        self.find_parent(node, self.root)
    
    def find_parent(self, node, parent):
        parent.value += node.value
        parent.max = max(node.value, parent.max)
        parent.min = min(node.value, parent.min)

        if parent.free:
            node.code = parent.code + [0]
            parent.left = node
        elif not parent.full:                
            if parent.left.value <= node.value:
                node.code = parent.code + [1]
                parent.right = node
            else:
                parent.left.code = parent.code + [1]
                parent.right = copy.deepcopy(parent.left)
                node.code = parent.code + [0]
                parent.left = node                    
        elif parent.left.leaf and parent.right.leaf:
            if parent.left.value <= node.value:
                if parent.right.value <= node.value:
                    left = copy.deepcopy(parent.left)
                    middle = copy.deepcopy(parent.right)
                    parent.right = node
                else:
                    left = copy.deepcopy(parent.left)
                    middle = node                        
            else:
                left = node
                middle = copy.deepcopy(parent.left)                    
            parent.left = IndexedNode(0, -1, left=left, right=middle, code=parent.code+[0])
        elif not (parent.left.leaf or parent.right.leaf):
            if node.value < parent.left.max:
                self.find_parent(node, parent.left)
            elif node.value >= parent.right.min:
                self.find_parent(node, parent.right)
            else:
                self.find_parent(node, parent.left) if np.random.rand() > 0.5 else self.find_parent(node, parent.right)
        elif not parent.left.leaf and parent.right.leaf:
            if node.value < parent.left.max:
                self.find_parent(node, parent.left)
            elif node.value >= parent.right.value:
                left = copy.deepcopy(parent.right)
                parent.right = IndexedNode(0, -1, left=left, right=node, code=parent.code+[1])
            else:
                right = copy.deepcopy(parent.right)
                parent.right = IndexedNode(0, -1, left=node, right=right, code=parent.code+[1])
        elif parent.left.leaf and not parent.right.leaf:
            if node.value >= parent.right.min:
                self.find_parent(node, parent.right)
            elif node.value < parent.left.value:
                right = copy.deepcopy(parent.left)
                parent.left = IndexedNode(0, -1, left=node, right=right, code=parent.code+[0])
            else:
                left = copy.deepcopy(parent.left)
                parent.left = IndexedNode(0, -1, left=left, right=node, code=parent.code+[0])
        else:
            assert 0==1, 'Error. Method: find_parent. Description: Case not considered.'
              
    def sample(self, batch_size):
        partition = np.linspace(0.0, self.root.value, num=batch_size)
        indices, codes = [], []
        return indices, codes


#-------------------------------------------------------------
#
#    Value networks
#
#-------------------------------------------------------------
class v_Net(nn.Module):
    def __init__(self, input_dim, n_tasks, lr=3e-4):
        super().__init__()        
        self.l1_E = nn.Linear(input_dim, 256)        
        self.l2_E = nn.Linear(256, 256)
        self.l3_E = nn.Linear(256, n_tasks)

        # self.l1_I = nn.Linear(input_dim, 256)
        # self.l2_I = nn.Linear(256, 256)        
        # self.l3_I = nn.Linear(256, n_tasks)  

        self.apply(weights_init_)       
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.bn = nn.BatchNorm1d(256)        
    
    def forward(self, s):
        x_E = F.relu(self.l1_E(s))
        x_E = F.relu(self.l2_E(x_E))
        V_E = self.l3_E(x_E)

        # x_I = F.relu(self.l1_I(s))
        # x_I = F.relu(self.l2_I(x_I))
        # V_I = self.l3_I(x_I)
        return V_E #, V_I

class q_Net(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks, lr=3e-4):
        super().__init__()        
        self.l1_E = nn.Linear(s_dim+a_dim, 256)
        self.l2_E = nn.Linear(256, 256)
        self.l3_E = nn.Linear(256, n_tasks)

        # self.l1_I = nn.Linear(s_dim+a_dim, 256)
        # self.l2_I = nn.Linear(256, 256)        
        # self.l3_I = nn.Linear(256, n_tasks)    

        self.apply(weights_init_) 
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)   

        self.bn = nn.BatchNorm1d(256) 
    
    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x_E = F.relu(self.l1_E(x))
        x_E = F.relu(self.l2_E(x_E))
        Q_E = self.l3_E(x_E)

        # x_I = F.relu(self.l1_I(x))
        # x_I = F.relu(self.l2_I(x_I))
        # Q_I = self.l3_I(x_I)
        return Q_E #, Q_I

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

        self.l11 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l12 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l21 = parallel_Linear(n_tasks, 256, 256)
        self.l22 = parallel_Linear(n_tasks, 256, 256)
        self.lV_E1 = parallel_Linear(n_tasks, 256, 1)
        self.lV_E2 = parallel_Linear(n_tasks, 256, 1)
        # self.lV_I = parallel_Linear(n_tasks, 256, 1)
        self.lA_E1 = parallel_Linear(n_tasks, 256, n_skills)
        self.lA_E2 = parallel_Linear(n_tasks, 256, n_skills) 
        # self.lA_I = parallel_Linear(n_tasks, 256, n_skills)  

        self.apply(weights_init_noisy)
        self.loss_func = nn.SmoothL1Loss() 
        self.optimizer = optim.Adam(self.parameters(), lr=lr)    
    
    def forward(self, s):        
        mu = self.l11(s)
        log_sigma = self.l12(s).clamp(-4.0,2.0)
        ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        mu = self.l21(x)
        log_sigma = self.l22(x).clamp(-4.0,2.0)
        ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        x = F.relu(mu + eij*torch.exp(log_sigma))

        muV = self.lV_E1(x)
        log_sigmaV = self.lV_E2(x).clamp(-4.0,2.0)
        eiV = torch.randn(muV.size(0), self.n_tasks, 1).to(device)
        ejV = torch.randn(1, self.n_tasks, muV.size(2)).to(device)
        eijV = torch.sign(eiV)*torch.sign(ejV)*(eiV).abs()**0.5*(ejV).abs()**0.5
        V_E = muV + eijV*torch.exp(log_sigmaV)

        muA = self.lA_E1(x)
        log_sigmaA = self.lA_E2(x).clamp(-4.0,2.0)
        eiA = torch.randn(muA.size(0), self.n_tasks, 1).to(device)
        ejA = torch.randn(1, self.n_tasks, muA.size(2)).to(device)
        eijA = torch.sign(eiA)*torch.sign(ejA)*(eiA).abs()**0.5*(ejA).abs()**0.5
        A_E = muA + eijA*torch.exp(log_sigmaA)

        # V_E = self.lV_E(x)
        # V_I = self.lV_I(x)
        # A_E = self.lA_E(x)
        # A_I = self.lA_I(x)
        Q_E = V_E + A_E - A_E.mean(2, keepdim=True) #.view(-1, self.n_tasks, self.n_skills)
        # Q_I = V_I + A_I - A_I.mean(2, keepdim=True)
        return Q_E #, Q_I 

    #     nc1 = vision_channels * 2
    #     nc2 = vision_channels * 4
    #     nc3 = vision_channels * 8
    #     nc4 = vision_channels * 16

    #     kernel_size1 = 4
    #     kernel_size2 = 4
    #     kernel_size3 = 3
    #     kernel_size4 = 3

    #     dilation1 = 1
    #     dilation2 = 2
    #     dilation3 = 1
    #     dilation4 = 2
        
    #     k_dim1 = 24
    #     k_dim2 = 20
    #     k_dim3 = 15
    #     k_dim4 = 10

    #     v_dim1 = int((vision_dim - dilation1*(kernel_size1-1) - 1)/1 + 1)
    #     v_dim2 = int((v_dim1 - dilation2*(kernel_size2-1) - 1)/1 + 1)
    #     v_dim3 = int((v_dim2 - dilation3*(kernel_size3-1) - 1)/1 + 1)
    #     v_dim4 = int((v_dim3 - dilation4*(kernel_size4-1) - 1)/1 + 1)
        
    #     self.lv1e = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
    #     self.lv2e = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
    #     self.lv3e = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
    #     self.lv4e = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)
    #     self.lv1g = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
    #     self.lv2g = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
    #     self.lv3g = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
    #     self.lv4g = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)        

    #     self.lk1 = multichannel_Linear(1, nc1, self.kinematic_dim, k_dim1)
    #     self.lk2 = multichannel_Linear(nc1, nc2, k_dim1, k_dim2)
    #     self.lk3 = multichannel_Linear(nc2, nc3, k_dim2, k_dim3)
    #     self.lk4 = multichannel_Linear(nc3, nc4, k_dim3, k_dim4)

    #     self.lc1x1 = nn.Conv1d(nc4, n_tasks, 1, stride=1)
    #     self.lkv = parallel_Linear(n_tasks, v_dim4+k_dim4, 256)
    #     self.lA_E = parallel_Linear(n_tasks, 256, n_skills)
    #     self.lA_I = parallel_Linear(n_tasks, 256, n_skills)
    #     self.lV_E = parallel_Linear(n_tasks, 256, 1)
    #     self.lV_I = parallel_Linear(n_tasks, 256, 1)
        
    #     self.bn1 = nn.BatchNorm1d(nc1)
    #     self.bn2 = nn.BatchNorm1d(nc2)
    #     self.bn3 = nn.BatchNorm1d(nc3)
    #     self.bn4 = nn.BatchNorm1d(nc4)
    #     self.bn5 = nn.BatchNorm1d(n_tasks)
                        
    #     self.apply(weights_init_)

    #     self.loss_func = nn.SmoothL1Loss(reduction='none')
    #     self.optimizer = optim.Adam(self.parameters(), lr=lr)       

    # def forward(self, s):
    #     vision_input = s[:,-int(self.vision_dim*self.vision_channels):].view(s.size(0),self.vision_channels,self.vision_dim)
    #     kinematic_input = s[:,:-int(self.vision_dim*self.vision_channels)].unsqueeze(1)

    #     v = torch.tanh(self.bn1(self.lv1e(vision_input))) * torch.sigmoid(self.bn1(self.lv1g(vision_input)))
    #     v = torch.tanh(self.bn2(self.lv2e(v))) * torch.sigmoid(self.bn2(self.lv2g(v)))
    #     v = torch.tanh(self.bn3(self.lv3e(v))) * torch.sigmoid(self.bn3(self.lv3g(v)))
    #     v = torch.tanh(self.bn4(self.lv4e(v))) * torch.sigmoid(self.bn4(self.lv4g(v)))
        
    #     k = F.relu(self.bn1(self.lk1(kinematic_input)))
    #     k = F.relu(self.bn2(self.lk2(k)))
    #     k = F.relu(self.bn3(self.lk3(k)))
    #     k = torch.tanh(self.bn4(self.lk4(k)))

    #     x = torch.cat([k,v],2)
    #     x = F.relu(self.bn5(self.lc1x1(x)))
    #     x = F.relu(self.bn5(self.lkv(x)))
    #     V_E = self.lV_E(x)
    #     V_I = self.lV_I(x)
    #     A_E = self.lA_E(x)
    #     A_I = self.lA_I(x)
    #     Q_E = V_E + A_E - A_E.mean(2, keepdim=True) #.view(-1, self.n_tasks, self.n_skills)
    #     Q_I = V_I + A_I - A_I.mean(2, keepdim=True)        
    #     return Q_E, Q_I  
  

class RND_subNet(nn.Module):
    def __init__(self, s_dim, out_dim, n_tasks, target=False):
        super().__init__()  
        self.s_dim = s_dim
        
        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, out_dim)

        self.apply(weights_init_big) if target else self.apply(weights_init_)        
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

class RND_Net(nn.Module):
    def __init__(self, s_dim, n_tasks, out_dim=20, lr=3e-4, alpha=1e-2):
        super().__init__() 
        self.s_dim = s_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.n_tasks = n_tasks
        
        self.target = RND_subNet(s_dim, out_dim, n_tasks, target=True)
        self.predictor = RND_subNet(s_dim, out_dim, n_tasks) 

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        self.mean_s = torch.zeros(n_tasks, s_dim).to(device)
        self.std_s = 1e-4*torch.ones(n_tasks, s_dim).to(device)
        self.std_e = torch.ones(n_tasks, 1).to(device)        
    
    def forward(self, s, t):
        t_one_hot = np.zeros([t.shape[0], self.n_tasks])
        t_one_hot[np.arange(t.shape[0]), t] = np.ones(t.shape[0])
        t_one_hot_distribution = torch.from_numpy(t_one_hot / (t_one_hot.sum(0, keepdims=True) + 1e-10)).float().to(device)

        self.mean_s = (1.0-self.alpha) * self.mean_s + self.alpha * (s.unsqueeze(1).detach() * t_one_hot_distribution.unsqueeze(2)).sum(0)
        self.std_s = (1.0-self.alpha) * self.std_s + self.alpha * ((((s.detach() - self.mean_s[t,:])**2).unsqueeze(1) * t_one_hot_distribution.unsqueeze(2)).sum(0))**0.5

        s_normalized = (s - self.mean_s[t,:]) / self.std_s[t,:]
        s_normalized = s_normalized.clamp(-5.0,5.0)

        noise = self.target(s_normalized)
        prediction = self.predictor(s_normalized)
        error = (((prediction - noise)[np.arange(t.shape[0]), t, :])**2).sum(1, keepdim=True)

        self.std_e = (1.0-self.alpha) * self.std_e + self.alpha * (((error.detach()**2).unsqueeze(1) * t_one_hot_distribution.unsqueeze(2)).sum(0))**0.5

        error_normalized = error / self.std_e[t,:]
        
        return(error_normalized) 

class d_Net(nn.Module):
    def __init__(self, s_dim, n_tasks, min_log_stdev=-4, max_log_stdev=2, lr=3e-4, alpha=0.99, beta=5.0e0, max_C=np.log(2), delta_C=np.log(2)*1.0e-5, C_0=0.0):
        super().__init__()  
        self.s_dim = s_dim
        self.n_tasks = n_tasks   
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev
        self.alpha = alpha
        self.beta = beta
        self.max_C = max_C
        self.delta_C = delta_C
        self.C = C_0
        
        self.l1 = parallel_Linear_simple(n_tasks, self.s_dim, (self.s_dim*3)//4)
        self.l2 = parallel_Linear(n_tasks, (self.s_dim*3)//4, (self.s_dim*1)//2)
        self.l31 = parallel_Linear(n_tasks, (self.s_dim*1)//2, (self.s_dim*1)//4)
        self.l32 = parallel_Linear(n_tasks, (self.s_dim*1)//2, (self.s_dim*1)//4)
        self.l4 = parallel_Linear(n_tasks, (self.s_dim*1)//4, (self.s_dim*1)//2)
        self.l5 = parallel_Linear(n_tasks, (self.s_dim*1)//2, (self.s_dim*3)//4)
        self.l6 = parallel_Linear(n_tasks, (self.s_dim*3)//4, self.s_dim)
        
        self.apply(weights_init_) 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.estimated_mean = torch.zeros(1,s_dim).to(device)
        self.estimated_std = torch.zeros(1,s_dim).to(device)
    
    def forward(self, s, t):
        x = torch.tanh(self.l1(s))
        x = torch.tanh(self.l2(x))
        mu_z = self.l31(x)
        log_sigma_z = self.l32(x)
        log_sigma_z = torch.clamp(log_sigma_z, self.min_log_stdev, self.max_log_stdev)

        ei = torch.randn(mu_z.size(0), self.n_tasks, 1).to(device)
        ej = torch.randn(1, self.n_tasks, mu_z.size(2)).to(device)
        eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        y = mu_z + eij*torch.exp(log_sigma_z)
        y = torch.tanh(self.l4(y))
        y = torch.tanh(self.l5(y))
        y = self.l6(y)

        self.estimated_mean = (self.alpha * self.estimated_mean + (1.0-self.alpha) * s.mean(0, keepdim=True)).detach()
        self.estimated_std = (self.alpha * self.estimated_std + (1.0-self.alpha) * ((s-self.estimated_mean)**2).mean(0, keepdim=True)**0.5).detach()
        
        return y[np.arange(t.shape[0]), t, :], mu_z[np.arange(t.shape[0]), t, :], log_sigma_z[np.arange(t.shape[0]), t, :]
    
    def loss_func(self, s, y, mu_z, log_sigma_z):
        posterior_error = (0.5*(mu_z**2 + torch.exp(log_sigma_z)**2 - 1) - log_sigma_z).sum(1).mean()
        reconstruction_error = (((s - y) / (self.estimated_std+1e-6))**2).sum(1).mean()
        loss = self.beta * (posterior_error - self.C).abs() + reconstruction_error

        self.C = np.min([self.C + self.delta_C, self.max_C])

        return loss

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
    def __init__(self, n_concepts, s_dim, n_skills, n_tasks=1, lr=3e-4, vision_dim=20, vision_channels=3):
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


