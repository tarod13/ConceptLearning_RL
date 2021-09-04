import math
import random
import copy
import numpy as np
import heapq as hq

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple) or isinstance(m, Linear):
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

def weights_init_om(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.orthogonal_(m.weight, 0.1)
        try:
            m.bias.data.zero_()
        except:
            pass

def weights_init_os(m):
    if isinstance(m, nn.Linear) or isinstance(m, parallel_Linear) or isinstance(m, parallel_Linear_simple):
        torch.nn.init.orthogonal_(m.weight, 0.01)
        try:
            m.bias.data.zero_()
        except:
            pass

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
        # set_seed(n_seed)
    
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

class VanillaPriorityQueue: 
    def __init__(self): 
        self.queue = [] 
  
    def __str__(self): 
        return ' '.join([str(i) for i in self.queue]) 
  
    @property
    def empty(self): 
        return len(self.queue) == 0 
  
    def insert(self, sample): 
        self.queue.append(sample) 
  
    def remove(self): 
        try: 
            max = 0
            for i in range(len(self.queue)): 
                if self.queue[i][0] > self.queue[max][0]: 
                    max = i 
            item = self.queue[max] 
            del self.queue[max] 
            return item 
        except IndexError: 
            print() 
            exit() 

class HeapPriorityQueue:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks 
        self.queue = []
        self.directory = {str(i):{} for i in range(0, n_tasks)}                
    
    @property
    def empty(self): 
        return len(self.queue) == 0

    def add(self, task, state, priority):
        add = True
        if str(state) in self.directory[str(task)]:
            old_priority, _, _ = self.directory[str(task)][str(state)]
            if priority < old_priority:
                self.remove(task, state)
            else:
                add = False
        if add:
            entry = [priority, task, state]
            self.directory[str(task)][str(state)] = entry
            hq.heappush(self.queue, entry)
    
    def remove(self, task, state):
        entry = self.directory[str(task)].pop(str(state))
        entry[-1] = -1
    
    def pop(self):
        while not self.empty:
            priority, task, state = hq.heappop(self.queue)
            if state != -1:
                del self.directory[str(task)][str(state)]
                return task, state
        return 0
        #raise KeyError('pop from an empty priority queue')


#-------------------------------------------------------------
#
#    Value networks
#
#-------------------------------------------------------------
# class v_Net(nn.Module):
#     def __init__(self, input_dim, n_tasks, lr=3e-4):
#         super().__init__()        
#         self.l1_E = nn.Linear(input_dim, 256)        
#         self.l2_E = nn.Linear(256, 256)
#         self.l3_E = nn.Linear(256, n_tasks)

#         # self.l1_I = nn.Linear(input_dim, 256)
#         # self.l2_I = nn.Linear(256, 256)        
#         # self.l3_I = nn.Linear(256, n_tasks)  

#         self.apply(weights_init_)       
#         self.loss_func = nn.MSELoss()
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)

#         self.bn = nn.BatchNorm1d(256)        
    
#     def forward(self, s):
#         x_E = F.relu(self.l1_E(s))
#         x_E = F.relu(self.l2_E(x_E))
#         V_E = self.l3_E(x_E)

#         # x_I = F.relu(self.l1_I(s))
#         # x_I = F.relu(self.l2_I(x_I))
#         # V_I = self.l3_I(x_I)
#         return V_E #, V_I

# class q_Net(nn.Module):
#     def __init__(self, s_dim, a_dim, n_tasks, lr=3e-4):
#         super().__init__()        
#         self.l1_E = nn.Linear(s_dim+a_dim, 256)
#         self.l2_E = nn.Linear(256, 256)
#         self.l3_E = nn.Linear(256, n_tasks)

#         # self.l1_I = nn.Linear(s_dim+a_dim, 256)
#         # self.l2_I = nn.Linear(256, 256)        
#         # self.l3_I = nn.Linear(256, n_tasks)    

#         self.apply(weights_init_) 
#         self.loss_func = nn.MSELoss()
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)   

#         self.bn = nn.BatchNorm1d(256) 
    
#     def forward(self, s,a):
#         x = torch.cat([s, a], 1)
#         x_E = F.relu(self.l1_E(x))
#         x_E = F.relu(self.l2_E(x_E))
#         Q_E = self.l3_E(x_E)

#         # x_I = F.relu(self.l1_I(x))
#         # x_I = F.relu(self.l2_I(x_I))
#         # Q_I = self.l3_I(x_I)
#         return Q_E #, Q_I
        
class v_Net(nn.Module):
    def __init__(self, input_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks        
        self.l11_E = parallel_Linear_simple(n_tasks, input_dim, 256)        
        self.l12_E = parallel_Linear(n_tasks, 256, 256)
        self.l13_E = parallel_Linear(n_tasks, 256, 1)

        # self.l21_E = parallel_Linear_simple(n_tasks, input_dim, 256)        
        # self.l22_E = parallel_Linear(n_tasks, 256, 256)
        # self.l23_E = parallel_Linear(n_tasks, 256, 1)

        # self.l1_I = nn.Linear(input_dim, 256)
        # self.l2_I = nn.Linear(256, 256)        
        # self.l3_I = nn.Linear(256, n_tasks)  

        self.apply(weights_init_)       
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.bn = nn.BatchNorm1d(256)        
    
    def forward(self, s):
        x_E = F.relu(self.l11_E(s))
        x_E = F.relu(self.l12_E(x_E))
        V_E = self.l13_E(x_E).squeeze(2)

        # x_I = F.relu(self.l1_I(s))
        # x_I = F.relu(self.l2_I(x_I))
        # V_I = self.l3_I(x_I)

        # mu = self.l11_E(s)
        # log_sigma = self.l21_E(s).clamp(-20.0,4.0)
        # ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        # ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        # eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        # x = F.relu(mu + eij*torch.exp(log_sigma))

        # mu = self.l12_E(x)
        # log_sigma = self.l22_E(x).clamp(-20.0,4.0)
        # ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        # ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        # eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        # x = F.relu(mu + eij*torch.exp(log_sigma))

        # mu = self.l13_E(x)
        # log_sigma = self.l23_E(x).clamp(-20.0,4.0)
        # ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        # ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        # eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        # V_E = (mu + eij*torch.exp(log_sigma)).squeeze(2)

        return V_E #, V_I

class q_Net(nn.Module):
    def __init__(self, s_dim, a_dim, n_tasks, lr=3e-4):
        super().__init__()      
        self.n_tasks = n_tasks  
        self.l11_E = parallel_Linear_simple(n_tasks, s_dim+a_dim, 256)
        self.l12_E = parallel_Linear(n_tasks, 256, 256)
        self.l13_E = parallel_Linear(n_tasks, 256, 1)

        # self.l21_E = parallel_Linear_simple(n_tasks, s_dim+a_dim, 256)        
        # self.l22_E = parallel_Linear(n_tasks, 256, 256)
        # self.l23_E = parallel_Linear(n_tasks, 256, 1)

        # self.l1_I = nn.Linear(s_dim+a_dim, 256)
        # self.l2_I = nn.Linear(256, 256)        
        # self.l3_I = nn.Linear(256, n_tasks)    

        self.apply(weights_init_) 
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)   

        self.bn = nn.BatchNorm1d(256) 
    
    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x_E = F.relu(self.l11_E(x))
        x_E = F.relu(self.l12_E(x_E))
        Q_E = self.l13_E(x_E).squeeze(2)

        # x_I = F.relu(self.l1_I(x))
        # x_I = F.relu(self.l2_I(x_I))
        # Q_I = self.l3_I(x_I)

        # mu = self.l11_E(x)
        # log_sigma = self.l21_E(x).clamp(-20.0,4.0)
        # ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        # ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        # eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        # x = F.relu(mu + eij*torch.exp(log_sigma))

        # mu = self.l12_E(x)
        # log_sigma = self.l22_E(x).clamp(-20.0,4.0)
        # ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        # ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        # eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        # x = F.relu(mu + eij*torch.exp(log_sigma))

        # mu = self.l13_E(x)
        # log_sigma = self.l23_E(x).clamp(-20.0,4.0)
        # ei = torch.randn(mu.size(0), self.n_tasks, 1).to(device)
        # ej = torch.randn(1, self.n_tasks, mu.size(2)).to(device)
        # eij = torch.sign(ei)*torch.sign(ej)*(ei).abs()**0.5*(ej).abs()**0.5
        # Q_E = (mu + eij*torch.exp(log_sigma)).squeeze(2)

        return Q_E #, Q_I

class m_Net(nn.Module):
    def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.n_concepts = n_concepts             
        self.l1 = parallel_Linear_empty(n_tasks * n_concepts, 64)
        self.l2 = parallel_Linear(n_tasks * n_concepts, 64, 64)
        self.l3 = parallel_Linear(n_tasks * n_concepts, 64, n_skills)
        self.l4 = nn.Softmax(dim=2)
        
        # self.bn1 = nn.BatchNorm1d(n_concepts+n_tasks) 
        # self.bn2 = nn.BatchNorm1d(256) 
        # self.bn3 = nn.BatchNorm1d(256) 

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    # def forward(self, S_one_hot, vars_=None):
    #     # x = torch.cat([S, T], 1)
    #     # x = F.relu(self.l1(self.bn1(x)))
    #     # x = F.relu(self.l2(self.bn2(x)))
    #     # x = self.l3(self.bn3(x))
    #     if vars_ is None:
    #         x = F.relu(self.l1(S_one_hot))
    #         x = F.relu(self.l2(x))
    #         x = (self.l3(x)).clamp(-20, 2) #.view(-1, self.n_tasks, self.n_skills)
    #     else:
    #         x = F.relu(self.l1(S_one_hot, vars_=vars_[0:2]))
    #         x = F.relu(self.l2(x, vars_=vars_[2:4]))
    #         x = (self.l3(x, vars_=vars_[4:])).clamp(-10, 2)

    #     PA_ST = self.l4(x)
    #     log_PA_ST = x - torch.logsumexp(x, dim=2, keepdim=True)
    #     return PA_ST, log_PA_ST
    
    def forward(self, vars_=None):
        x = F.relu(self.l1())
        x = F.relu(self.l2(x))
        x = (self.l3(x)).clamp(-20, 2).squeeze(0).view(self.n_tasks, self.n_concepts, self.n_skills)
        
        PA_ST = self.l4(x)
        log_PA_ST = x - torch.logsumexp(x, dim=2, keepdim=True)
        return PA_ST, log_PA_ST
    
class dm_Net(nn.Module):
    def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
        super().__init__()
        self.n_skills = n_skills
        self.n_tasks = n_tasks
        self.n_concepts = n_concepts        
        self.l1 = parallel_Linear_simple(n_tasks*n_concepts, n_skills, 64)
        self.l2 = parallel_Linear(n_tasks*n_concepts, 64, 64)
        self.l3 = parallel_Linear(n_tasks*n_concepts, 64, n_concepts)
        self.l4 = nn.Softmax(dim=3)

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    # def forward(self, S_one_hot, A_one_hot):
    def forward(self, A_one_hot):
        # x = torch.cat([S_one_hot, A_one_hot], 1)            
        x = F.relu(self.l1(A_one_hot))
        x = F.relu(self.l2(x))
        x = (self.l3(x)).clamp(-20, 2).view(-1,self.n_tasks,self.n_concepts,self.n_concepts)       
        PnS_STdoA = self.l4(x)
        log_PnS_STdoA = x - torch.logsumexp(x, dim=3, keepdim=True)
        return PnS_STdoA, log_PnS_STdoA

class dmT_Net(nn.Module):
    def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_skills = n_skills        
        self.l1 = parallel_Linear_simple(n_tasks, n_concepts+n_skills, 64)
        self.l2 = parallel_Linear(n_tasks, 64, 64)
        self.l3 = parallel_Linear(n_tasks, 64, n_concepts)
        self.l4 = nn.Softmax(dim=2)

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    def forward(self, S_one_hot, A_one_hot, vars_=None):
        x = torch.cat([S_one_hot, A_one_hot], 1)

        if vars_ is None:            
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = (self.l3(x)).clamp(-6, 6)
        else:
            x = F.relu(self.l1(x, vars_=vars_[0:2]))
            x = F.relu(self.l2(x, vars_=vars_[2:4]))
            x = (self.l3(x, vars_=vars_[4:])).clamp(-10, 2)

        PnS_SAT = self.l4(x)
        log_PnS_SAT = x - torch.logsumexp(x, dim=2, keepdim=True)
        return PnS_SAT, log_PnS_SAT

class cl_Net(nn.Module):
    def __init__(self, n_concepts, s_dim, n_tasks, lr=3e-4, vision_channels=3, vision_dim=20):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_concepts = n_concepts 

        self.l1 = Linear(s_dim, 256)
        self.l2 = Linear(256, 256)
        self.l3 = Linear(256, n_concepts)
        self.l4 = nn.Softmax(dim=1)
        
        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        
    def forward(self, s, vars_=None):   
        if vars_ is None:         
            x = F.relu(self.l1(s))
            x = F.relu(self.l2(x))
            x = (self.l3(x)) #.clamp(-6, 6)
        else:
            x = F.relu(self.l1(s, vars_=vars_[0:2]))
            x = F.relu(self.l2(x, vars_=vars_[2:4]))
            x = (self.l3(x, vars_=vars_[4:])) #.clamp(-20, 4)

        PS_s = self.l4(x)
        log_PS_s = x - torch.logsumexp(x, dim=1, keepdim=True)
        return PS_s, log_PS_s   

    #     self.vision_channels = vision_channels
    #     self.vision_dim = vision_dim
        
    #     nc1 = vision_channels * 2
    #     nc2 = vision_channels * 4
    #     nc3 = vision_channels * 8
    #     nc4 = vision_channels * 16
    #     nc5 = vision_channels * 32
    #     nc6 = vision_channels * 64

    #     kernel_size1 = 3
    #     kernel_size2 = 3
    #     kernel_size3 = 3
    #     kernel_size4 = 2
    #     kernel_size5 = 2
    #     kernel_size6 = 2
        
    #     dilation1 = 1
    #     dilation2 = 1
    #     dilation3 = 1
    #     dilation4 = 1
    #     dilation5 = 1
    #     dilation6 = 1
        
    #     v_dim1 = int((vision_dim - dilation1*(kernel_size1-1) - 1)/1 + 1)
    #     v_dim2 = int((v_dim1 - dilation2*(kernel_size2-1) - 1)/1 + 1)
    #     v_dim3 = int((v_dim2 - dilation3*(kernel_size3-1) - 1)/1 + 1)
    #     v_dim4 = int((v_dim3 - dilation4*(kernel_size4-1) - 1)/1 + 1)
    #     v_dim5 = int((v_dim4 - dilation5*(kernel_size5-1) - 1)/1 + 1)
    #     v_dim6 = int((v_dim5 - dilation6*(kernel_size6-1) - 1)/1 + 1)
        
    #     self.lv1e = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
    #     self.lv2e = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
    #     self.lv3e = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
    #     self.lv4e = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)
    #     self.lv1g = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
    #     self.lv2g = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
    #     self.lv3g = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
    #     self.lv4g = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)        

    #     self.lc1x1_1 = nn.Conv1d(nc1, nc1, 1, stride=1)
    #     self.lc1x1_2 = nn.Conv1d(nc2, nc2, 1, stride=1)
    #     self.lc1x1_3 = nn.Conv1d(nc3, nc3, 1, stride=1)
    #     self.lc1x1_4 = nn.Conv1d(nc4, nc4, 1, stride=1)
    #     self.lc1x1_5 = nn.Conv1d(nc4, n_concepts, 1, stride=1)
    #     self.lSM = nn.Softmax(dim=1)
        
    #     self.apply(weights_init_)
    #     self.optimizer = optim.Adam(self.parameters(), lr=lr)       

    # def forward(self, s):
    #     v = s[:,-int(self.vision_dim*self.vision_channels):].view(s.size(0),self.vision_channels,self.vision_dim)    
    #     v = torch.tanh(self.lv1e(v)) * torch.sigmoid(self.lv1g(v))
    #     v = self.lc1x1_1(v)
    #     v = torch.tanh(self.lv2e(v)) * torch.sigmoid(self.lv2g(v))
    #     v = self.lc1x1_2(v)
    #     v = torch.tanh(self.lv3e(v)) * torch.sigmoid(self.lv3g(v))
    #     v = self.lc1x1_3(v)
    #     v = F.relu(torch.tanh(self.lv4e(v)) * torch.sigmoid(self.lv4g(v)))
    #     v = F.relu(self.lc1x1_4(v))
    #     v = self.lc1x1_5(v).reshape(s.size(0),-1)

    #     PS_s = self.lSM(v)
    #     log_PS_s = v - torch.logsumexp(v, dim=1, keepdim=True)
    #     return PS_s, log_PS_s

class vS_Net(nn.Module):
    def __init__(self, n_concepts, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        
        self.l1 = parallel_Linear_simple(n_tasks, 1, 64)
        self.l2 = parallel_Linear(n_tasks, 64, 64)
        self.l3 = parallel_Linear(n_tasks, 64, n_concepts)
        self.l4 = nn.Softmax(dim=2)        

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    def forward(self, v):
        x = F.relu(self.l1(v))
        x = F.relu(self.l2(x))
        x = self.l3(x).clamp(-20, 2)

        PS_vT = self.l4(x)
        log_PS_vT = x - torch.logsumexp(x, dim=2, keepdim=True)
        return PS_vT, log_PS_vT

class A_Net(nn.Module):
    def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        
        self.l1 = parallel_Linear_simple(n_tasks, n_skills, 64)
        self.l2 = parallel_Linear(n_tasks, 64, 64)
        self.l3 = parallel_Linear(n_tasks, 64, n_concepts)
        self.l4 = nn.Softmax(dim=2)        

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    def forward(self, A_one_hot):
        x = F.relu(self.l1(A_one_hot))
        x = F.relu(self.l2(x))
        x = self.l3(x).clamp(-6, 6)

        PS_AT = self.l4(x)
        log_PS_AT = x - torch.logsumexp(x, dim=2, keepdim=True)
        return PS_AT, log_PS_AT

class r_Net(nn.Module):
    def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_skills = n_skills 

        self.l1 = parallel_Linear_simple(n_tasks, 2*n_concepts+n_skills, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 16)        

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    def r_modes(self, S_one_hot, A_one_hot, nS_one_hot):
        x = torch.cat([S_one_hot, A_one_hot, nS_one_hot], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def forward(self, S_one_hot, A_one_hot, r, nS_one_hot, T, vars_=None):
        r_modes = self.r_modes(S_one_hot, A_one_hot, nS_one_hot)[np.arange(r.shape[0]), T, :]
        modes = ((r_modes - r.view(-1,1))**2).argmin(1)
        r_approx = r_modes[np.arange(r.shape[0]), modes]
        return r_approx

class r_marginal_Net(nn.Module):
    def __init__(self, n_concepts, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks 

        self.l1 = parallel_Linear_simple(n_tasks, n_concepts, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 16)        

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
    def r_modes(self, nS_one_hot):
        x = F.relu(self.l1(nS_one_hot))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def forward(self, r, nS_one_hot, T, vars_=None):
        r_modes = self.r_modes(nS_one_hot)[np.arange(r.shape[0]), T, :]
        modes = ((r_modes - r.view(-1,1))**2).argmin(1)
        r_approx = r_modes[np.arange(r.shape[0]), modes]
        return r_approx

# class r_Net(nn.Module):
#     def __init__(self, n_concepts, n_skills, n_tasks, lr=3e-4):
#         super().__init__()
#         self.n_tasks = n_tasks
#         self.n_skills = n_skills 

#         self.l1_e = nn.Linear(1, 20)
#         self.l21_e = nn.Linear(20, 1)
#         self.l22_e = nn.Linear(20, 1)

#         self.l1_d = parallel_Linear_simple(n_tasks, 2*n_concepts+n_skills+1, 256)
#         self.l2_d = parallel_Linear(n_tasks, 256, 256)
#         self.l3_d = parallel_Linear(n_tasks, 256, 1)        

#         self.apply(weights_init_)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
#     def encode(self, r):
#         x = F.relu(self.l1_e(r.view(-1,1)))
#         mu = self.l21_e(x)
#         log_sigma = self.l22_e(x).clamp(-20,2)
#         return mu, log_sigma, torch.exp(log_sigma)
    
#     def decode(self, code, S_one_hot, A_one_hot, nS_one_hot):
#         x = torch.cat([code, S_one_hot, A_one_hot, nS_one_hot], 1)
#         x = F.relu(self.l1_d(x))
#         x = F.relu(self.l2_d(x))
#         x = self.l3_d(x)
#         return x

#     def forward(self, S_one_hot, A_one_hot, r, nS_one_hot, vars_=None):
#         mu, log_sigma, sigma = self.encode(r)
#         code = mu + sigma * torch.randn_like(mu)
#         r_approx = self.decode(code, S_one_hot, A_one_hot, nS_one_hot)
#         return r_approx, mu, log_sigma, sigma

# class r_marginal_Net(nn.Module):
#     def __init__(self, n_concepts, n_tasks, lr=3e-4):
#         super().__init__()
#         self.n_tasks = n_tasks
        
#         self.l1_e = nn.Linear(1, 20)
#         self.l21_e = nn.Linear(20, 1)
#         self.l22_e = nn.Linear(20, 1)

#         self.l1_d = parallel_Linear_simple(n_tasks, n_concepts+1, 256)
#         self.l2_d = parallel_Linear(n_tasks, 256, 256)
#         self.l3_d = parallel_Linear(n_tasks, 256, 1)        

#         self.apply(weights_init_)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
#     def encode(self, r):
#         x = F.relu(self.l1_e(r.view(-1,1)))
#         mu = self.l21_e(x)
#         log_sigma = self.l22_e(x).clamp(-20,2)
#         return mu, log_sigma, torch.exp(log_sigma)
    
#     def decode(self, code, nS_one_hot):
#         x = torch.cat([code, nS_one_hot], 1)
#         x = F.relu(self.l1_d(x))
#         x = F.relu(self.l2_d(x))
#         x = self.l3_d(x)
#         return x

#     def forward(self, r, nS_one_hot, vars_=None):
#         mu, log_sigma, sigma = self.encode(r)
#         code = mu + sigma * torch.randn_like(mu)
#         r_approx = self.decode(code, nS_one_hot)
#         return r_approx, mu, log_sigma, sigma

class softmax_actorNet(nn.Module):
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
        # log_PA_sT = x - torch.logsumexp(x, dim=2, keepdim=True)
        # log_PA_sT = torch.log(PA_sT + 1e-10)
        return PA_sT, log_PA_sT

class softmax_alphaNet(nn.Module):
    def __init__(self, s_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 1)
        
        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)
        
    def forward(self, s):    
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        log_alpha = (self.l3(x) - np.log(100)).clamp(-20,2).view(-1, self.n_tasks, 1)    
        return torch.exp(log_alpha), log_alpha

class attentionNet(nn.Module):
    def __init__(self, s_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l2 = parallel_Linear(n_tasks, 256, 256)
        self.l3 = parallel_Linear(n_tasks, 256, 1)
        
        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)
        
    def forward(self, s):    
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        lambda_ = torch.sigmoid(self.l3(x)).view(-1, self.n_tasks, 1)    
        return lambda_

class discrete_AC_Net(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=3e-4, wi=2.0, df_wi=0.9999, min_wi=1e-3):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        self.wi = wi
        self.min_wi = min_wi
        self.df_wi = df_wi
        
        self.critic1 = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)        
        self.critic1_target = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.critic2 = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.critic2_target = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.actor = softmax_actorNet(n_skills, s_dim, n_tasks, lr=lr)
        self.q_intrinsic = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.q_intrinsic_target = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        # self.alpha = softmax_alphaNet(s_dim, n_tasks)
        self.log_alpha = torch.zeros(n_tasks, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        updateNet(self.critic1_target, self.critic1, 1.0)
        updateNet(self.critic2_target, self.critic2, 1.0)
        updateNet(self.q_intrinsic_target, self.q_intrinsic, 1.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def q_values(self, s):
        q1 = self.critic1(s)
        q2 = self.critic2(s)
        qi = self.q_intrinsic(s)
        alpha, log_alpha = self.alpha(s)
        return q1, q2, qi, alpha, log_alpha

    def skill_lklhood(self, s, q1, q2, qi, alpha):
        qe = torch.min(torch.stack([q1, q2]), 0)[0]
        qt = qe + qi*self.wi        
        log_pi = qt/(alpha + 1e-10)
        log_pi = log_pi - torch.logsumexp(log_pi, dim=2, keepdim=True)
        pi = torch.exp(log_pi)
        return pi, log_pi

    def forward(self, s, T):
        q1, q2, qi, alpha, _ = self.q_values(s)
        q1T, q2T, qiT = q1[np.arange(s.size(0)), T, :], q2[np.arange(s.size(0)), T, :], qi[np.arange(s.size(0)), T, :]
        q1T_target = self.critic1_target(s)[np.arange(s.size(0)), T, :]
        q2T_target = self.critic2_target(s)[np.arange(s.size(0)), T, :]
        qiT_target = self.q_intrinsic_target(s)[np.arange(s.size(0)), T, :]
        # pi, log_pi = self.actor(s)       

        pi, log_pi = self.skill_lklhood(s, q1, q2, qi, alpha)
        piT, log_piT = pi[np.arange(s.size(0)), T, :], log_pi[np.arange(s.size(0)), T, :]

        return q1T, q2T, q1T_target, q2T_target, piT, log_piT, qiT, qiT_target
    
    def sample_skill(self, s, task, explore=True):
        # PA_sT = self.actor(s.view(1,-1))[0].squeeze(0)[task,:].view(-1)
        q1, q2, qi, alpha, _ = self.q_values(s.view(1,-1))
        PA_sT = self.skill_lklhood(s.view(1,-1), q1, q2, qi, alpha)[0].squeeze(0)[task,:].view(-1)
        if explore:
            A = Categorical(probs=PA_sT).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().item()            
        return A
    
    def sample_skills(self, s, T, explore=True):
        # PA_sT = self.actor(s)[0][np.arange(s.shape[0]), T, :]  
        q1, q2, qi, alpha, _ = self.q_values(s)
        PA_sT = self.skill_lklhood(s, q1, q2, qi, alpha)[0][np.arange(s.shape[0]), T, :]             
        if explore:
            A = Categorical(probs=PA_sT).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            A = Categorical(probs=tie_breaking_dist).sample().cpu()                  
        return A
    
    def update_targets(self, lr):
        updateNet(self.critic1_target, self.critic1, lr)
        updateNet(self.critic2_target, self.critic2, lr)
        updateNet(self.q_intrinsic_target, self.q_intrinsic, lr)
        self.wi = np.max([self.min_wi, self.wi*self.df_wi])

class discrete_AC_Net_PG(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, n_concepts, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        
        self.qe1 = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe1_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qe2 = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe2_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi1_exploration = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi1_exploration_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi2_exploration = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi2_exploration_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi1_consensus = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi1_consensus_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi2_consensus = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi2_consensus_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.actor = softmax_actorNet(n_skills, s_dim, n_tasks, lr=lr)        
        self.lambda_ = attentionNet(s_dim, n_tasks)

        # self.alpha = softmax_alphaNet(s_dim, n_tasks, lr=lr)
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
        updateNet(self.qi1_exploration_target, self.qi1_exploration, 1.0)
        updateNet(self.qi2_exploration_target, self.qi2_exploration, 1.0)
        updateNet(self.qi1_consensus_target, self.qi1_consensus, 1.0)
        updateNet(self.qi2_consensus_target, self.qi2_consensus, 1.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s, T):
        qe1 = self.qe1(s)[np.arange(s.size(0)), T, :]
        qe1_target = self.qe1_target(s)[np.arange(s.size(0)), T, :]
        qe2 = self.qe2(s)[np.arange(s.size(0)), T, :]
        qe2_target = self.qe2_target(s)[np.arange(s.size(0)), T, :]
        qi1_exp = self.qi1_exploration(s)[np.arange(s.size(0)), T, :]
        qi1_exp_target = self.qi1_exploration_target(s)[np.arange(s.size(0)), T, :]
        qi2_exp = self.qi2_exploration(s)[np.arange(s.size(0)), T, :]
        qi2_exp_target = self.qi2_exploration_target(s)[np.arange(s.size(0)), T, :]
        qi1_con = self.qi1_consensus(s)[np.arange(s.size(0)), T, :]
        qi1_con_target = self.qi1_consensus_target(s)[np.arange(s.size(0)), T, :]
        qi2_con = self.qi2_consensus(s)[np.arange(s.size(0)), T, :]
        qi2_con_target = self.qi2_consensus_target(s)[np.arange(s.size(0)), T, :]
        pi, log_pi = self.actor(s)
        pi, log_pi = pi[np.arange(s.size(0)), T, :], log_pi[np.arange(s.size(0)), T, :]        
        # alpha, log_alpha = self.alpha(s)
        # alpha, log_alpha = alpha.view(-1,1), log_alpha.view(-1,1)
        alpha, log_alpha = self.alpha[T].view(-1,1), self.log_alpha[T].view(-1,1)
        Alpha, log_Alpha = self.Alpha, self.log_Alpha
        lambda_ = self.lambda_(s).view(-1,1)
        return qe1, qe1_target, qe2, qe2_target, qi1_exp, qi1_exp_target, qi2_exp, qi2_exp_target, qi1_con, qi1_con_target, qi2_con, qi2_con_target, pi, log_pi, alpha, log_alpha, lambda_, Alpha, log_Alpha
    
    def sample_skill(self, s, task, explore=True, rng=None):
        PA_sT = self.actor(s.view(1,-1))[0].squeeze(0)[task,:].view(-1)
        # A = Categorical(probs=PA_sT).sample().item()
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
        # updateNet(self.qi1_exploration_target, self.qi1_exploration, lr)
        # updateNet(self.qi2_exploration_target, self.qi2_exploration, lr)
        # updateNet(self.qi1_consensus_target, self.qi1_consensus, lr)
        # updateNet(self.qi2_consensus_target, self.qi2_consensus, lr)

class discrete_AC_Net_PG_simple(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        
        self.qe1 = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe1_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qe2 = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe2_target = DuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        
        self.actor = softmax_actorNet(n_skills, s_dim, n_tasks, lr=lr)        
        self.lambda_ = attentionNet(s_dim, n_tasks)

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

class DQN_actor_Net(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=3e-4, eps=5e-2):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        self.epsilon = eps
        
        self.qe = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)        
        self.qe_target = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi_exploration = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi_exploration_target = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi_consensus = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        self.qi_consensus_target = NoisyDuelingDQN(s_dim, n_skills, n_tasks, lr=lr)
        
        updateNet(self.qe_target, self.qe, 1.0)
        updateNet(self.qi_exploration_target, self.qi_exploration, 1.0)
        updateNet(self.qi_consensus_target, self.qi_consensus, 1.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def q_values(self, s):
        qe = self.qe(s)
        qi_exp = self.qi_exploration(s)
        qi_con = self.qi_consensus(s)
        return qe, qi_exp, qi_con

    def forward(self, s, T):
        qe, qi_exp, qi_con = self.q_values(s)
        qeT, qiT_exp, qiT_con = qe[np.arange(s.size(0)), T, :], qi_exp[np.arange(s.size(0)), T, :], qi_con[np.arange(s.size(0)), T, :]
        qeT_target = self.qe_target(s)[np.arange(s.size(0)), T, :]
        qiT_exp_target = self.qi_exploration_target(s)[np.arange(s.size(0)), T, :]
        qiT_con_target = self.qi_consensus_target(s)[np.arange(s.size(0)), T, :]
        return qeT, qeT_target, qiT_exp, qiT_exp_target, qiT_con, qiT_con_target
    
    def sample_skill(self, s, task, explore=True):
        qe, qi_exp, qi_con = self.q_values(s.view(1,-1))
        qe, qi_exp, qi_con = qe.squeeze(0)[task, :], qi_exp.squeeze(0)[task,:], qi_con.squeeze(0)[task,:]
        qt = qe + qi_exp + qi_con
        assert torch.all(qt==qt)
        if explore and np.random.rand() < self.epsilon:
            A = np.random.randint(self.n_skills)
        else:
            tie_breaking_dist = torch.isclose(qt, qt.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().item()
        return A
    
    def sample_skills(self, s, T):
        qe, qi_exp, qi_con = self.q_values(s)
        qe, qi_exp, qi_con = qe[np.arange(s.size(0)), T , :], qi_exp[np.arange(s.size(0)), T , :], qi_con[np.arange(s.size(0)), T , :]
        qt = qe + qi_exp + qi_con                    
        tie_breaking_dist = torch.isclose(qt, qt.max(1, keepdim=True)[0]).float()
        tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
        A = Categorical(probs=tie_breaking_dist).sample().cpu()                  
        return A
    
    def update_targets(self, lr):
        updateNet(self.qe_target, self.qe, lr)
        updateNet(self.qi_exploration_target, self.qi_exploration, lr)
        updateNet(self.qi_consensus_target, self.qi_consensus, lr)
        
class PPO_discrete_actorNet(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=3e-4):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 

        self.l1 = parallel_Linear_simple(n_tasks, s_dim, 64)
        self.l2 = parallel_Linear(n_tasks, 64, 64)
        self.l3 = parallel_Linear(n_tasks, 64, n_skills)
        self.l4 = nn.Softmax(dim=2)
        
        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, s):    
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)    
        PA_sT = self.l4(x)
        # log_PA_sT = x - torch.logsumexp(x, dim=1, keepdim=True)
        log_PA_sT = torch.log(PA_sT + 1e-10)
        return PA_sT, log_PA_sT

class PPO_discreteNet(nn.Module):
    def __init__(self, n_skills, s_dim, n_tasks, lr=1e-2):
        super().__init__()
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_skills = n_skills 
        
        self.critic = v_Net(s_dim, n_tasks, lr=lr)
        self.critic_target = v_Net(s_dim, n_tasks, lr=lr)
        self.actor = PPO_discrete_actorNet(n_skills, s_dim, n_tasks, lr=lr)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, s, A, T):
        v = self.critic(s)[np.arange(s.size(0)), T].view(-1,1)
        v_target = self.critic_target(s)[np.arange(s.size(0)), T].view(-1,1)
        pi, log_pi = self.actor(s)
        entropy = -(pi[np.arange(s.size(0)), T, :] * log_pi[np.arange(s.size(0)), T, :]).sum(1)                
        return v, v_target, log_pi[np.arange(s.size(0)), T, A], entropy
    
    def sample_skill(self, s, task, explore=True):
        PA_sT = self.actor(s.view(1,-1))[0].squeeze(0)[task,:].view(-1)
        if explore:
            A = Categorical(probs=PA_sT).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(PA_sT, PA_sT.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().item()            
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

class DuelingDQN(nn.Module):
    def __init__(self, s_dim, n_skills, n_tasks, vision_dim=20, vision_channels=3, lr=3e-4):
        super().__init__()  
        self.s_dim = s_dim
        self.n_skills = n_skills
        self.n_tasks = n_tasks   
        self.vision_dim = vision_dim
        self.vision_channels = vision_channels
        self.kinematic_dim = s_dim - vision_dim*vision_channels 

        self.l11 = parallel_Linear_simple(n_tasks, s_dim, 256)
        self.l21 = parallel_Linear(n_tasks, 256, 256)
        self.lV_E1 = parallel_Linear(n_tasks, 256, 1)
        self.lA_E1 = parallel_Linear(n_tasks, 256, n_skills)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV_E1.weight, 0.01)
        self.lV_E1.bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA_E1.weight, 0.01)
        self.lA_E1.bias.data.zero_()

        self.loss_func = nn.SmoothL1Loss() 
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-4)    
    
    def forward(self, s):        
        mu = F.relu(self.l11(s))
        mu = F.relu(self.l21(mu))
        V_E = self.lV_E1(mu)        
        A_E = self.lA_E1(mu)
        Q_E = V_E + A_E - A_E.mean(2, keepdim=True) 
        return Q_E

class NoisyDuelingDQN(nn.Module):
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

        muV = self.lV_E1(x)
        log_sigmaV = self.lV_E2(x).clamp(-20.0,2.0)
        eiV = torch.randn(muV.size(0), self.n_tasks, 1).to(device)
        ejV = torch.randn(1, self.n_tasks, muV.size(2)).to(device)
        eijV = torch.sign(eiV)*torch.sign(ejV)*(eiV).abs()**0.5*(ejV).abs()**0.5
        V_E = muV + eijV*torch.exp(log_sigmaV)

        muA = self.lA_E1(x)
        log_sigmaA = self.lA_E2(x).clamp(-20.0,2.0)
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
        # new_var = new_sum_of_squared_diff / new_count
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

class RND_Net(nn.Module):
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
        # self.q_rms = RunningMeanStd(n_tasks, 1, N=1000, init_mean=1.0)
        # self.q_rms = RunningMeanStd(n_tasks, 1, N = 10, init_mean=1.0)
        # self.discounted_reward = RewardForwardFilter(self.gamma_I)

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
class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, vars_=None):
        if vars_ is None:
            weight = self.weight
            bias = self.bias
        else:
            weight, bias = vars_
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

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

class parallel_Linear_empty(nn.Module):
    def __init__(self, n_layers, out_features):
        super().__init__()
        self.out_features = out_features
        self.n_layers = n_layers
        self.weight = Parameter(torch.Tensor(n_layers, out_features))      
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, vars_=None):
        if vars_ is None:
            weight = self.weight            
        else:
            weight = vars_
        return weight.unsqueeze(0) 

    def extra_repr(self):
        return 'out_features={}'.format(
            self.out_features is not None
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
    def __init__(self, n_concepts, s_dim, n_skills, n_tasks=1, lr=3e-4, tau=1.0):
        super().__init__()  
        self.n_tasks = n_tasks
        self.s_dim = s_dim   
        self.n_concepts = n_concepts
        self.tau = tau   
        
        self.classifier = cl_Net(n_concepts, s_dim, n_tasks)
        self.map = m_Net(n_concepts, n_skills, n_tasks)
        self.model = dm_Net(n_concepts, n_skills, n_tasks)
        # self.v_model = vS_Net(n_concepts, n_tasks)
        # self.A_model = A_Net(n_concepts, n_skills, n_tasks)
        self.ly = nn.Softmax(dim=1)

        self.apply(weights_init_)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # params_model = list(self.map.parameters()) + list(self.model.parameters()) + list(self.v_model.parameters()) + list(self.A_model.parameters())
        # self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        # self.model_optimizer = optim.Adam(params_model, lr=lr)

    
    def sample_concept(self, s, explore=True):
        PS_s = self.classifier(s.view(1,-1))[0].view(-1)
        if explore:
            S = Categorical(probs=PS_s).sample().item()
        else:            
            tie_breaking_dist = torch.isclose(PS_s, PS_s.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            S = Categorical(probs=tie_breaking_dist).sample().item()            
        return S, PS_s

    def sample_differentiable_concepts(self, PS_s, log_PS_s):
        u = torch.rand_like(PS_s)
        g = -torch.log(-torch.log(u+1e-10)+1e-10)
        e = g + log_PS_s
        z = torch.zeros_like(PS_s)
        z[np.arange(0,PS_s.shape[0]), e.argmax(1)] = torch.ones(PS_s.shape[0]).to(device)
        y = self.ly(e/self.tau)
        z = z + (y - y.detach()).clamp(0.0,1.0)
        return z

    # def sample_skill(self, s, task, explore=True):
    #     S, PS = self.sample_concept(s, explore=explore)
    #     z = self.sample_differentiable_concepts(PS)
    #     PA_ST, _ = self.map(z)[:,task,:]
    #     if explore:
    #         A = Categorical(probs=PA_ST).sample().item()
    #     else:            
    #         tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max()).float()
    #         tie_breaking_dist /= tie_breaking_dist.sum()
    #         A = Categorical(probs=tie_breaking_dist).sample().item()
    #     return A, PA_ST, S, PS
    
    def sample_concepts(self, s, explore=True, vars_=None):
        PS_s, log_PS_s = self.classifier(s) #, vars_=vars_)
        if explore:
            S = Categorical(probs=PS_s).sample().cpu()
        else:            
            tie_breaking_dist = torch.isclose(PS_s, PS_s.max(1, keepdim=True)[0]).float()
            tie_breaking_dist /= tie_breaking_dist.sum(1, keepdim=True)
            S = Categorical(probs=tie_breaking_dist).sample().cpu()   
        return S, PS_s, log_PS_s  

    def forward(self, s, A_one_hot, ns, STE=True, explore=False):
        S, PS_s, log_PS_s = self.sample_concepts(s, explore=explore)
        nS, PnS_ns, log_PnS_ns = self.sample_concepts(ns, explore=explore)

        PA_ST, log_PA_ST = self.map()
        PnS_STdoA, log_PnS_STdoA = self.model(A_one_hot)

        # if STE:
        #     # z = PS_s
        #     z = self.sample_differentiable_concepts(PS_s, log_PS_s)
        #     # nz = self.sample_differentiable_concepts(PnS_ns, log_PnS_ns)
        # else:
        #     z = torch.zeros_like(PS_s)
        #     z[np.arange(0,PS_s.shape[0]), S] = torch.ones(PS_s.shape[0]).to(device)
        #     # z[np.arange(0,PS_s.shape[0]), PS_s.argmax(1)] = torch.ones(PS_s.shape[0]).to(device)
        #     # z = self.sample_differentiable_concepts(PS_s.detach(), log_PS_s.detach())
        #     # nz = self.sample_differentiable_concepts(PnS_ns.detach(), log_PnS_ns.detach())
        
        # PA_ST, log_PA_ST = self.map(z)
        # PnS_STdoA, log_PnS_STdoA = self.model(z, A_one_hot)
        # PS_vT, log_PS_vT = self.v_model(v)
        # PS_AT, log_PS_AT = self.A_model(A_off_one_hot)

        # tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max(2, keepdim=True)[0]).float()
        # tie_breaking_dist /= tie_breaking_dist.sum()
        # A = Categorical(probs=tie_breaking_dist).sample()

        return PS_s, log_PS_s, PnS_ns, log_PnS_ns, PA_ST, log_PA_ST, PnS_STdoA, log_PnS_STdoA #, A, PS_vT, log_PS_vT, PS_AT, log_PS_AT
        

# class c_Net(nn.Module):
#     def __init__(self, n_concepts, s_dim, n_skills, n_tasks=1, lr=3e-4, vision_dim=20, vision_channels=3, tau=2.0, lr_sch_steps=5000, gamma=0.6, N=5, meta=False):
#         super().__init__()  
#         self.n_tasks = n_tasks
#         self.s_dim = s_dim   
#         self.n_concepts = n_concepts
#         self.tau = tau   
        
#         self.classifier = cl_Net(n_concepts, s_dim, n_tasks)
#         self.map = m_Net(n_concepts, n_skills, n_tasks)
#         self.model = dm_Net(n_concepts, n_skills, n_tasks)
#         self.r_model = r_Net(n_concepts, n_skills, n_tasks)
#         self.r_marginal_model = r_marginal_Net(n_concepts, n_tasks)
#         self.ly = nn.Softmax(dim=1)

#         if meta:
#             self.alpha = nn.ParameterList([])
#             self.alpha_op1 = nn.ParameterList([])
#             self.alpha_op2 = nn.ParameterList([])
#             for i in range(3*3*2):
#                 alpha = Parameter(torch.ones(N))           
#                 self.alpha.append(alpha)
#                 if i < 12:
#                     self.alpha_op1.append(alpha)
#                 if i < 6 or i >= 12:
#                     self.alpha_op2.append(alpha)            
#             for alpha in self.alpha: nn.init.constant_(alpha, lr)
        
#         self.apply(weights_init_)
#         params_model = list(self.map.parameters()) + list(self.model.parameters()) + list(self.r_model.parameters()) + list(self.r_marginal_model.parameters())
#         self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
#         self.model_optimizer = optim.Adam(params_model, lr=lr)
#         # self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, step_size=lr_sch_steps, gamma=gamma)

#     # def forward(self, s):   
#     #     x = F.relu(self.l1(s))
#     #     x = F.relu(self.l2(x))
#     #     x = self.l3(x)
#     #     PS_s = self.l4(x)
#     #     log_PS_s = x - torch.logsumexp(x, dim=1, keepdim=True)
#     #     return PS_s, log_PS_s
    
#     def sample_concept(self, s, explore=True):
#         PS_s = self.classifier(s.view(1,-1))[0].view(-1)
#         if explore:
#             S = Categorical(probs=PS_s).sample().item()
#         else:            
#             tie_breaking_dist = torch.isclose(PS_s, PS_s.max()).float()
#             tie_breaking_dist /= tie_breaking_dist.sum()
#             S = Categorical(probs=tie_breaking_dist).sample().item()            
#         return S, PS_s

#     def sample_differentiable_concepts(self, PS_s, log_PS_s):
#         u = torch.rand_like(PS_s)
#         # p_unnormalized = torch.log(u+1e-10)
#         # p = p_unnormalized / p_unnormalized.sum(1, keepdim=True)
#         # p_improved = p - PS_s + PS_s.detach()
#         g = -torch.log(-torch.log(u+1e-10)+1e-10)
#         e = g + log_PS_s
#         # assert torch.all(e == e), 'EXPLOSION e!'
#         # e = log_PS_s.detach() - torch.log(p_improved + 1e-10)
#         z = torch.zeros_like(PS_s)
#         z[np.arange(0,PS_s.shape[0]), e.argmax(1)] = torch.ones(PS_s.shape[0]).to(device)
#         y = self.ly(e/self.tau)
#         # assert torch.all(y == y), 'EXPLOSION y!'
#         z = z + (y - y.detach()).clamp(0.0,1.0)
#         # self.tau = np.max([0.99999 * self.tau, 0.1])
#         return z

#     def sample_skill(self, s, task, explore=True):
#         S, PS = self.sample_concept(s, explore=explore)
#         z = self.sample_differentiable_concepts(PS)
#         PA_ST, _ = self.map(z)[:,task,:]
#         if explore:
#             A = Categorical(probs=PA_ST).sample().item()
#         else:            
#             tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max()).float()
#             tie_breaking_dist /= tie_breaking_dist.sum()
#             A = Categorical(probs=tie_breaking_dist).sample().item()
#         return A, PA_ST, S, PS
    
#     def sample_concepts(self, s, explore=True, vars_=None):
#         PS_s, log_PS_s = self.classifier(s, vars_=vars_)
#         if explore:
#             S = Categorical(probs=PS_s).sample().cpu()
#         else:            
#             tie_breaking_dist = torch.isclose(PS_s, PS_s.max(1, keepdim=True)[0]).float()
#             tie_breaking_dist /= tie_breaking_dist.sum()
#             S = Categorical(probs=tie_breaking_dist).sample().cpu()            
#         return S, PS_s, log_PS_s  

#     def forward(self, s, A_one_hot, r, ns, T, active_model=True, active_map=True, vars_cla=None, vars_map=None, vars_mod=None, STE=True):
#         S, PS_s, log_PS_s = self.sample_concepts(s, explore=False, vars_=vars_cla)
#         nS, PnS_ns, log_PnS_ns = self.sample_concepts(ns, explore=False, vars_=vars_cla)

#         if STE:
#             z = self.sample_differentiable_concepts(PS_s, log_PS_s)
#             nz = self.sample_differentiable_concepts(PnS_ns, log_PnS_ns)
#         else:
#             z = torch.zeros_like(PS_s).to(device)
#             z[np.arange(0,PS_s.shape[0]), PS_s.argmax(1)] = torch.ones(PS_s.shape[0]).to(device)
#             nz = torch.zeros_like(PS_s).to(device)
#             nz[np.arange(0,PS_s.shape[0]), PnS_ns.argmax(1)] = torch.ones(PS_s.shape[0]).to(device)
#         # if STE:
#         #     z = self.sample_differentiable_concepts(PS_s, log_PS_s)
#         #     nz = self.sample_differentiable_concepts(PnS_ns, log_PnS_ns)
#         # else:
#         #     z = self.sample_differentiable_concepts(PS_s.detach(), log_PS_s.detach())
#         #     nz = self.sample_differentiable_concepts(PnS_ns.detach(), log_PnS_ns.detach())

#         if active_map: 
#             PA_ST, log_PA_ST = self.map(z, vars_=vars_map)
#             tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max(2, keepdim=True)[0]).float()
#             tie_breaking_dist /= tie_breaking_dist.sum()
#             A = Categorical(probs=tie_breaking_dist).sample()

#         if active_model: 
#             PnS_SAT, log_PnS_SAT = self.model(z, A_one_hot, vars_=vars_mod)
#             # r_SAT, mu_SAT, log_sigma_SAT, sigma_SAT = self.r_model(z, A_one_hot, r, nz)   
#             # r_nST, mu_nST, log_sigma_nST, sigma_nST = self.r_marginal_model(r, nz)
#             r_SAT = self.r_model(z, A_one_hot, r, nz, T)   
#             r_nST = self.r_marginal_model(r, nz, T)

#         if active_model and active_map:
#             # return S, PS_s, log_PS_s, nS, PnS_ns, log_PnS_ns, A, PA_ST, log_PA_ST, PnS_SAT, log_PnS_SAT, r_SAT, mu_SAT, log_sigma_SAT, sigma_SAT, r_nST, mu_nST, log_sigma_nST, sigma_nST
#             return S, PS_s, log_PS_s, nS, PnS_ns, log_PnS_ns, A, PA_ST, log_PA_ST, PnS_SAT, log_PnS_SAT, r_SAT, r_nST
#         elif active_map:
#             return S, PS_s, log_PS_s, nS, PnS_ns, log_PnS_ns, A, PA_ST, log_PA_ST
#         elif active_model:
#             # return S, PS_s, log_PS_s, nS, PnS_ns, log_PnS_ns, PnS_SAT, log_PnS_SAT, r_SAT, mu_SAT, log_sigma_SAT, sigma_SAT, r_nST, mu_nST, log_sigma_nST, sigma_nST
#             return S, PS_s, log_PS_s, nS, PnS_ns, log_PnS_ns, PnS_SAT, log_PnS_SAT, r_SAT, r_nST
#         else:
#             return S, PS_s, log_PS_s, nS, PnS_ns, log_PnS_ns
   
    # def sample_skills(self, s, explore=True):
    #     S, PS_s, log_PS_s = self.sample_concepts(s, explore=explore)
    #     z = self.sample_differentiable_concepts(log_PS_s)
    #     # assert torch.all(z == z), 'EXPLOSION z!'
    #     PA_ST, log_PA_ST = self.map(z)
    #     # assert torch.all(PA_ST == PA_ST), 'EXPLOSION map!'
    #     # if explore:
    #     #     A = Categorical(probs=PA_ST).sample().cpu()
    #     # else:            
    #     #     tie_breaking_dist = torch.isclose(PA_ST, PA_ST.max(2, keepdim=True)[0]).float()
    #     #     tie_breaking_dist /= tie_breaking_dist.sum()
    #     #     A = Categorical(probs=tie_breaking_dist).sample().cpu()
    #     return PA_ST, log_PA_ST, S, PS_s, log_PS_s, z

    # self.n_tasks = n_tasks
        # self.s_dim = s_dim   
        # self.n_concepts = n_concepts
        # self.vision_channels = vision_channels
        # self.vision_dim = vision_dim
        # self.kinematic_dim = s_dim - vision_dim*vision_channels  
        # self.tau = tau   
        
    #     nc1 = vision_channels * 2
    #     nc2 = vision_channels * 4
    #     nc3 = vision_channels * 8
    #     # nc4 = vision_channels * 16

    #     kernel_size1 = 4
    #     kernel_size2 = 4
    #     kernel_size3 = 3
    #     # kernel_size4 = 3

    #     dilation1 = 1
    #     dilation2 = 2
    #     dilation3 = 1
    #     # dilation4 = 2
        
    #     k_dim1 = 256
    #     # k_dim2 = 256
    #     # k_dim3 = 10
    #     # k_dim4 = 10

    #     v_dim1 = int((vision_dim - dilation1*(kernel_size1-1) - 1)/1 + 1)
    #     v_dim2 = int((v_dim1 - dilation2*(kernel_size2-1) - 1)/1 + 1)
    #     v_dim3 = int((v_dim2 - dilation3*(kernel_size3-1) - 1)/1 + 1)
    #     # v_dim4 = int((v_dim3 - dilation4*(kernel_size4-1) - 1)/1 + 1)
        
    #     self.lv1e = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
    #     self.lv2e = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
    #     self.lv3e = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
    #     # self.lv4e = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)
    #     self.lv1g = nn.Conv1d(vision_channels, nc1, kernel_size1, dilation=dilation1)
    #     self.lv2g = nn.Conv1d(nc1, nc2, kernel_size2, dilation=dilation2)
    #     self.lv3g = nn.Conv1d(nc2, nc3, kernel_size3, dilation=dilation3)
    #     # self.lv4g = nn.Conv1d(nc3, nc4, kernel_size4, dilation=dilation4)        

    #     self.lk1 = nn.Linear(self.kinematic_dim, k_dim1)
    #     # self.lk2 = nn.Linear(k_dim1, k_dim2)
    #     # self.lk1 = multichannel_Linear(1, nc1, self.kinematic_dim, k_dim1)
    #     # self.lk2 = multichannel_Linear(nc1, nc2, k_dim1, k_dim2)
    #     # self.lk3 = multichannel_Linear(nc2, nc3, k_dim2, k_dim3)
    #     # self.lk4 = multichannel_Linear(nc3, nc4, k_dim3, k_dim4)

    #     self.lc1x1 = nn.Conv1d(nc3, 1, 1, stride=1)
    #     self.lkv = nn.Linear(v_dim3+k_dim1, 256)
    #     self.lkv2 = nn.Linear(256, n_concepts)
    #     # self.lkv = parallel_Linear(n_concepts, v_dim3+k_dim1, 1)
    #     # self.lkv2 = nn.Linear(n_concepts, n_concepts)
        
    #     # self.bnv1 = nn.BatchNorm1d(vision_channels)
    #     # self.bnv2 = nn.BatchNorm1d(nc1)
    #     # self.bnv3 = nn.BatchNorm1d(nc2)
    #     # # self.bnv4 = nn.BatchNorm1d(nc3)
    #     # self.bnv1g = nn.BatchNorm1d(vision_channels)
    #     # self.bnv2g = nn.BatchNorm1d(nc1)
    #     # self.bnv3g = nn.BatchNorm1d(nc2)
    #     # # self.bnv4g = nn.BatchNorm1d(nc3)
    #     # self.bnk1 = nn.BatchNorm1d(self.kinematic_dim)
    #     # self.bnk2 = nn.BatchNorm1d(nc1)
    #     # self.bnk3 = nn.BatchNorm1d(nc2)
    #     # # self.bnk4 = nn.BatchNorm1d(nc3)

    #     # self.bn4 = nn.BatchNorm1d(nc3)

    #     self.map = m_Net(n_concepts, n_skills, n_tasks)
                        
    #     self.apply(weights_init_)

    #     self.optimizer = optim.Adam(self.parameters(), lr=lr)        

    # def forward(self, s):
    #     vision_input = s[:,-int(self.vision_dim*self.vision_channels):].view(s.size(0),self.vision_channels,self.vision_dim)
    #     kinematic_input = s[:,:-int(self.vision_dim*self.vision_channels)]

    #     # v = torch.tanh(self.lv1e(self.bnv1(vision_input))) * torch.sigmoid(self.lv1g(self.bnv1g(vision_input)))
    #     # v = torch.tanh(self.lv2e(self.bnv2(v))) * torch.sigmoid(self.lv2g(self.bnv2g(v)))
    #     # v = torch.tanh(self.lv3e(self.bnv3(v))) * torch.sigmoid(self.lv3g(self.bnv3g(v)))
    #     # # v = torch.tanh(self.lv4e(self.bnv4(v))) * torch.sigmoid(self.lv4g(self.bnv4g(v)))
        
    #     # k = F.relu(self.lk1(self.bnk1(kinematic_input).unsqueeze(1)))
    #     # k = F.relu(self.lk2(self.bnk2(k)))
    #     # k = F.relu(self.lk3(self.bnk3(k)))
    #     # # k = torch.tanh(self.lk4(self.bnk4(k)))

    #     # x = torch.cat([k,v],2)
    #     # x = F.relu(self.lc1x1(self.bn4(x)))
    #     # x = torch.sigmoid(self.lkv(x).squeeze(2))
    #     # x = self.lkv2(x)
    #     # x = torch.exp(x - x.max(1, keepdim=True)[0])
    #     # PS_s = x / x.sum(1, keepdim=True)
    #     # assert torch.all(PS_s == PS_s), 'EXPLOSION PS!'

    #     v = torch.tanh(self.lv1e(vision_input)) * torch.sigmoid(self.lv1g(vision_input))
    #     v = torch.tanh(self.lv2e(v)) * torch.sigmoid(self.lv2g(v))
    #     v = torch.tanh(self.lv3e(v)) * torch.sigmoid(self.lv3g(v))
    #     v = F.relu(self.lc1x1(v)).squeeze(1)
        
    #     k = F.relu(self.lk1(kinematic_input))

    #     x = torch.cat([k,v],1)        
    #     x = torch.sigmoid(self.lkv(x))
    #     x = self.lkv2(x)
    #     x = torch.exp(x - x.max(1, keepdim=True)[0])
    #     PS_s = x / x.sum(1, keepdim=True)
    #     assert torch.all(PS_s == PS_s), 'EXPLOSION PS!'

    #     return PS_s
    
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

        # if self.latent_dim > 0:
        #     t = torch.randn(1, self.latent_dim).float().to(device)
        #     x = torch.cat([x,t], 1)
        # x = F.relu(self.l11.conditional(x, A))
        # x = F.relu(self.l21.conditional(x, A))
        
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

        # x = s.clone()
        # if self.latent_dim > 0:
        #     t = torch.randn(s.size(0), 1, self.latent_dim).repeat(1,self.n_m_actions,1).float().cuda()
        #     x = torch.cat([x,t], 2)
        # x1 = F.relu(self.l11(x))
        # x1 = F.relu(self.l21(x1))

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


