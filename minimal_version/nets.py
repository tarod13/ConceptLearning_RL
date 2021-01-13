import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter

from custom_layers import parallel_Linear, Linear_noisy
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


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


# Simple convolutional net that outputs a feature vector
class vision_Net(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.apply(weights_init_rnd)
        
    def forward(self, x):
        conv_feat = self.conv(x).view(x.size()[0], -1)
        return conv_feat


class noisy_dueling_q_Net(nn.Module):
    def __init__(self, s_dim, n_actions, lr=3e-4):
        super().__init__() 
        self.s_dim = s_dim
        self.n_actions = n_actions

        self.l1 = Linear_noisy(s_dim, 256)
        self.l2 = Linear_noisy(256, 256)
        self.lV = Linear_noisy(256, 1)
        self.lA = Linear_noisy(256, n_actions)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV.weight, 0.01)
        self.lV_E1.bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA.weight, 0.01)
        self.lA_E1.bias.data.zero_()
        
    def forward(self, s):        
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        V = self.lV(mu)        
        A = self.lA(mu)
        Q = V.view(-1,1) + A - A.mean(1, keepdim=True) 
        return Q

# TODO: self.actor, self.lambda_, self.alpha
class discrete_actor_critic_Net(nn.Module):
    def __init__(self, s_dim, n_actions):
        super().__init__()   

        self.s_dim = s_dim
        self.n_actions = n_actions     

        self.q1 = noisy_dueling_q_Net(s_dim, n_actions, n_tasks, lr=lr)        
        self.q1_target = noisy_dueling_q_Net(s_dim, n_actions, n_tasks, lr=lr)
        self.q2 = noisy_dueling_q_Net(s_dim, n_actions, n_tasks, lr=lr)        
        self.q2_target = noisy_dueling_q_Net(s_dim, n_actions, n_tasks, lr=lr)
        
        self.actor = softmax_actorNet(n_actions, s_dim, n_tasks, lr=lr)        
        self.lambda_ = attentionNet(s_dim, n_tasks)

        self.log_alpha = torch.zeros(n_tasks).to(device)
        self.log_alpha.requires_grad = True
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        updateNet(self.q1_target, self.q1, 1.0)
        updateNet(self.q2_target, self.q2, 1.0)
    
    def forward(self, s):
        q1 = self.q1(s)
        q1_target = self.q1_target(s)
        q2 = self.q2(s)
        q2_target = self.q2_target(s)
        pi, log_pi = self.actor(s)
        alpha, log_alpha = self.alpha.view(-1,1), self.log_alpha.view(-1,1)
        return q1, q1_target, q2, q2_target, pi, log_pi, alpha, log_alpha
    
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
                A = rng.choice(self.n_actions, p=PA_sT.detach().cpu().numpy())
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
        updateNet(self.q1_target, self.q1, lr)
        updateNet(self.q2_target, self.q2, lr)


class vision_actor_critic_Net(nn.Module):
    def __init__(self, n_actions, s_dim, lr=3e-4):
        super().__init__()

        self.vision_net = vision_Net()
        self.vision_net_target = vision_Net()
        self.actor_critic_net = discrete_actor_critic_Net(n_actions, s_dim + 256)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, inner_state, outer_state):
        vision_features = self.vision_net(outer_state)
        observation = torch.cat([inner_state, vision_features])
        output = self.actor_critic_net(observation)
        return output