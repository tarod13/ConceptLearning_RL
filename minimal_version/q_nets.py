import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from custom_layers import Linear_noisy, parallel_Linear
from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class q_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__() 
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.lQ = nn.Linear(256, 1)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lQ.weight, 0.01)
        self.lQ.bias.data.zero_()
        
    def forward(self, s, a):
        sa = torch.cat([s,a], dim=1)        
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        Q = self.lQ(x)        
        return Q


class noisy_q_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__() 
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.l1 = Linear_noisy(s_dim+a_dim, 256)
        self.l2 = Linear_noisy(256, 256)
        self.lQ = Linear_noisy(256, 1)
        
        torch.nn.init.orthogonal_(self.lQ.mean_weight, 0.01)
        self.lQ.mean_bias.data.zero_()
        
    def forward(self, s, a):
        sa = torch.cat([s,a], dim=1)        
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        Q = self.lQ(x)        
        return Q


class dueling_q_Net(nn.Module):
    def __init__(self, s_dim, n_actions):
        super().__init__() 
        self.s_dim = s_dim
        self.n_actions = n_actions

        self.l1 = nn.Linear(s_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.lV = nn.Linear(256, 1)
        self.lA = nn.Linear(256, n_actions)
        
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


class multihead_dueling_q_Net(nn.Module):
    def __init__(self, s_dim, n_actions, n_heads):
        super().__init__() 
        self.s_dim = s_dim
        self.n_actions = n_actions

        self.l1 = parallel_Linear(n_heads, s_dim, 256)
        # self.l2 = parallel_Linear(n_heads, 256, 256)
        self.lV = parallel_Linear(n_heads, 256, 1)
        self.lA = parallel_Linear(n_heads, 256, n_actions)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV.weight, 0.01)
        self.lV.bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA.weight, 0.01)
        self.lA.bias.data.zero_()
        
    def forward(self, s):        
        x = F.relu(self.l1(s))
        #x = F.relu(self.l2(x))
        V = self.lV(x)        
        A = self.lA(x)
        Q = V + A - A.mean(2, keepdim=True) 
        return Q


class vision_multihead_dueling_q_Net(multihead_dueling_q_Net):
    def __init__(self, s_dim, latent_dim, n_actions, n_heads, lr=1e-4):
        super().__init__(s_dim + latent_dim, n_actions, n_heads)
        self.vision_nets = nn.ModuleList([vision_Net(latent_dim=latent_dim, 
            noisy=False) for i in range(0, n_heads)])
        self._n_heads = n_heads
        
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, inner_state, outer_state):    
        state = []
        for head in range(0, self._n_heads):
            head_features = self.vision_nets[head](outer_state)
            state.append(torch.cat([inner_state, head_features], dim=1))
        state = torch.stack(state, dim=1)
        x = F.relu(self.l1(state))
        V = self.lV(x)        
        A = self.lA(x)
        Q = V + A - A.mean(2, keepdim=True) 
        return Q


class noisy_dueling_q_Net(nn.Module):
    def __init__(self, s_dim, n_actions):
        super().__init__() 
        self.s_dim = s_dim
        self.n_actions = n_actions

        self.l1 = Linear_noisy(s_dim, 256)
        # self.l2 = Linear_noisy(256, 256)
        self.lV = Linear_noisy(256, 1)
        self.lA = Linear_noisy(256, n_actions)
        
        # self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV.mean_weight, 0.01)
        self.lV.mean_bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA.mean_weight, 0.01)
        self.lA.mean_bias.data.zero_()
        
    def forward(self, s):        
        x = F.relu(self.l1(s))
        # x = F.relu(self.l2(x))
        V = self.lV(x)        
        A = self.lA(x)
        Q = V.view(-1,1) + A - A.mean(1, keepdim=True) 
        return Q


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