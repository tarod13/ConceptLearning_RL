import gym
import torch
import numpy as np
from scipy.special import gamma as f_gamma

import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.nn.utils import clip_grad_norm_

from nets import (Memory, v_valueNet, q_valueNet, policyNet, conditionalEncoder, mixtureConceptModel, encoderConceptModel, 
                    rNet, rewardNet, v_parallel_valueNet, q_parallel_valueNet, nextSNet, conceptLossNet)
from tabularQ import Agent as mAgent

import os
import time
import pickle
from sys import stdout
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from torch.utils.tensorboard import SummaryWriter

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image

width = 1024
height = 768
FPS = 60

fourcc = VideoWriter_fourcc(*'MP42')

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

plt.ion()
# writer = SummaryWriter()
###########################################################################
#
#                           General methods
#
###########################################################################
def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def normalize_angle(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def scale_action(a, min, max):
    return (0.5*(a+1.0)*(max-min) + min)

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes).to(device) 
    return y[labels]

def set_seed(n_seed):
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)

def intersection(list1, list2): 
    return list(set(list1) & set(list2))

def vectorized_multinomial(prob_matrix):
    items = np.arange(prob_matrix.shape[1])
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0],1)
    k = (s < r).sum(axis=1)
    return items[k]

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=20) # zoom in on the lower gradient regions
    plt.ylim(bottom = -0.001, top=10)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show(block=False)
    plt.pause(0.3)

###########################################################################
#
#                               Classes
#
###########################################################################
#-------------------------------------------------------------
#
#                          Agent class
#
#-------------------------------------------------------------
class Agent:
    def __init__(self, s_dim, a_dim, params):

        self.params = params.copy()
        default_params = {
                            'hierarchical': False, 
                            'n_m_actions': 1, 
                            'n_m_states': 1,
                            'n_tasks': 2,
                            'embedded_envs': False,
                            'skill_steps': 9,
                            'concept_steps': 1,
                            'parallel_learning': False,
                            'reward_learning': 'with_concept',
                            'alpha_upper_level': 10.0*np.ones([1,1]),
                            'delta_upper_level': 1e-1,
                            'eta_upper_level': 1e-1,
                            'mu_upper_level': 1e-1,
                            'min_delta_upper_level': 3e-4,
                            'min_eta_upper_level': 3e-4,
                            'min_mu_upper_level': 3e-4,
                            'rate_delta_upper_level': 1.0,
                            'upper_level_period': 3, 
                            'concept_model_type': 'encoder',
                            'policy_batch_size': 256,
                            'concept_batch_size': 256,  
                            'soft_lr': 5e-3, 
                            'memory_capacity': 500000, 
                            'gamma': 0.99, 
                            'alpha': 1.0,
                            'tau_alpha': 3e-6, 
                            'tau_mu': 3e-4, 
                            'beta': 1.0, 
                            'eta': 1.0, 
                            'mu': 1.0, 
                            'nu': 1.0,
                            'zeta': 1.0,
                            'xi': 1.0, 
                            'theta': 0.1,
                            'hidden_dim': 256,
                            'min_log_stdev': -20, 
                            'max_log_stdev': 2,
                            'q_lr': 3e-4, 
                            'v_lr': 3e-4, 
                            'cm_lr': 3e-4, 
                            'p_lr': 3e-4, 
                            'r_lr': 3e-4, 
                            'state_norm': False,                 
                            'verbose': False, 
                            'seed': 1000,
                            'min_c_s': 25, 
                            'llhood_samples': 128, 
                            'clip_value': 1,
                            'concept_latent_dim': 0,
                            'vNets_parallel': True,
                            'policy_latent_dim': 0,
                            'inconsistency_metric': 'poly',
                            'upper_level_annealing' : True,
                            'model_update_method': 'discrete',
                            'value_update_method': 'discrete',
                            'tau_upper_level': 1.0,
                            'coefficient_upper_annealing': 0.999993,
                            'min_alpha_upper_level': 5e-2,                            
                            'threshold_entropy_alpha_upper_level': 0.69,
                            'threshold_entropy_beta_upper_level': 1.39,
                            'threshold_entropy_alpha': 0.69,
                            'threshold_entropy_mu': 5.2,
                            'beta_upper_level': 1.0,
                            'alpha_upper_level_threshold': 10.0,
                            'beta_upper_level_threshold': 10.0,
                            'alpha_threshold': 10.0,
                            'mu_threshold': 10.0,
                            'upper_policy_steps': 100,
                            'cancel_beta_upper_level': False,
                            'alpha_bias': 5e-2,
                            'automatic_lower_temperature': False                            
                        }
        
        for key, value in default_params.items():
            if key not in self.params.keys():
                self.params[key] = value
        
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = 2*self.s_dim + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 3
        self.N_s_dim = (self.params['upper_level_period']+1)*s_dim        

        # Hierarchical vars
        self.n_m_actions = self.params['n_m_actions']
        self.n_m_states = self.params['n_m_states']
        self.n_tasks = self.params['n_tasks']
        self.upper_level_period = self.params['upper_level_period']
        self.alpha_upper_level = self.params['alpha_upper_level']
        self.beta_upper_level = self.params['beta_upper_level']
        self.delta_upper_level = self.params['delta_upper_level']
        self.eta_upper_level = self.params['eta_upper_level']
        self.mu_upper_level = self.params['mu_upper_level']
        self.min_delta_upper_level = self.params['min_delta_upper_level']
        self.min_eta_upper_level = self.params['min_eta_upper_level']
        self.min_mu_upper_level = self.params['min_mu_upper_level']
        self.rate_delta_upper_level = self.params['rate_delta_upper_level']
        self.min_alpha_upper_level = self.params['min_alpha_upper_level']
        self.gamma = self.params['gamma']
        self.skill_steps = self.params['skill_steps']
        self.concept_steps = self.params['concept_steps']
        self.parallel_learning = self.params['parallel_learning']
        self.reward_learning = self.params['reward_learning']
        self.inconsistency_metric = self.params['inconsistency_metric']
        self.upper_level_annealing = self.params['upper_level_annealing']
        self.coefficient_upper_annealing = self.params['coefficient_upper_annealing']
        self.model_update_method = self.params['model_update_method']
        self.value_update_method = self.params['value_update_method']
        self.tau_upper_level = self.params['tau_upper_level']
        self.threshold_entropy_alpha = self.params['threshold_entropy_alpha']
        self.threshold_entropy_mu = self.params['threshold_entropy_mu']
        self.threshold_entropy_alpha_upper_level = self.params['threshold_entropy_alpha_upper_level']
        self.threshold_entropy_beta_upper_level = self.params['threshold_entropy_beta_upper_level']
        self.alpha_upper_level_threshold = self.params['alpha_upper_level_threshold']
        self.beta_upper_level_threshold = self.params['beta_upper_level_threshold']
        self.alpha_threshold = self.params['alpha_threshold']
        self.mu_threshold = self.params['mu_threshold']
        self.upper_policy_steps = self.params['upper_policy_steps']
        self.cancel_beta_upper_level = self.params['cancel_beta_upper_level']
        self.alpha_bias = self.params['alpha_bias']

        # State machine vars to handle the time period of the upper level
        self.m_state = 0
        self.m_action = 0
        self.state = []
        self.past_m_state = 0
        self.past_m_action = 0
        self.past_states = []
        self.past_actions = []
        self.action_llhoods = np.zeros((1, self.n_m_actions))        
        self.cumulative_reward = 0.0        
        self.bottom_level_epsds = 0
        self.past_posterior = np.ones(self.n_m_states)/self.n_m_states
        self.posterior = np.ones(self.n_m_states)/self.n_m_states
        self.stored = False        

        # Model options
        self.vNets_parallel = self.params['vNets_parallel']
        self.hierarchical = self.params['hierarchical']        
        self.concept_model_type = self.params['concept_model_type']
        self.embedded_envs = self.params['embedded_envs']
        self.multitask = self.n_tasks > 0 or self.embedded_envs
        self.automatic_lower_temperature = self.params['automatic_lower_temperature']

        # Net parameters
        self.policy_batch_size = self.params['policy_batch_size']
        self.concept_batch_size = self.params['concept_batch_size']
        self.soft_lr = self.params['soft_lr']
        self.clip_value = self.params['clip_value']
        
        # Metric weights
        if isinstance(self.params['alpha'], float):
            self.alpha = self.params['alpha'] * torch.ones(self.n_tasks).float().to(device)
        else:
            self.alpha = torch.from_numpy(self.params['alpha']).float().to(device)
        self.beta = self.params['beta']        
        self.eta = self.params['eta']
        self.nu = self.params['nu']
        if isinstance(self.params['mu'], float):
            self.mu = self.params['mu'] * torch.ones(self.n_tasks).float().to(device)
        else:
            self.mu = torch.from_numpy(self.params['mu']).float().to(device)        
        self.zeta = self.params['zeta']
        self.xi = self.params['xi']
        self.theta = self.params['theta']   
        self.tau_alpha = self.params['tau_alpha']
        self.tau_mu = self.params['tau_mu']     
        
        self.memory = Memory(self.params['memory_capacity'], n_seed=self.params['seed'])
        if self.vNets_parallel:
            self.critic1 = q_parallel_valueNet(s_dim, a_dim, n_tasks=self.n_tasks, lr=self.params['q_lr']).to(device)
            self.critic2 = q_parallel_valueNet(s_dim, a_dim, n_tasks=self.n_tasks, lr=self.params['q_lr']).to(device)
            self.baseline = v_parallel_valueNet(s_dim, n_tasks=self.n_tasks, lr=self.params['v_lr']).to(device)
            self.baseline_target = v_parallel_valueNet(s_dim, n_tasks=self.n_tasks, lr=self.params['v_lr']).to(device)
        else:
            self.critic1 = q_valueNet(s_dim, a_dim, n_tasks=self.n_tasks, lr=self.params['q_lr']).to(device)
            self.critic2 = q_valueNet(s_dim, a_dim, n_tasks=self.n_tasks, lr=self.params['q_lr']).to(device)
            self.baseline = v_valueNet(s_dim, n_tasks=self.n_tasks, lr=self.params['v_lr']).to(device)
            self.baseline_target = v_valueNet(s_dim, n_tasks=self.n_tasks, lr=self.params['v_lr']).to(device)
        self.actor = policyNet(self.n_m_actions, s_dim, a_dim, lr=self.params['p_lr'], hidden_dim=self.params['hidden_dim'], latent_dim=self.params['policy_latent_dim']).to(device)

        if self.hierarchical:
            if self.concept_model_type == 'mixture':
                self.concept_model = mixtureConceptModel(self.n_m_states, self.params['concept_latent_dim'], s_dim, llhood_samples=self.params['llhood_samples'], lr=self.params['cm_lr'], 
                                        state_norm=self.params['state_norm'], min_log_stdev=self.params['min_log_stdev'], max_log_stdev=self.params['max_log_stdev'], 
                                        min_c=self.params['min_c_s']).to(device)
            elif self.concept_model_type == 'encoder':
                self.concept_model = encoderConceptModel(self.n_m_states, s_dim, n_tasks=self.n_tasks, lr=self.params['cm_lr'], min_log_stdev=self.params['min_log_stdev'], 
                                        max_log_stdev=self.params['max_log_stdev'], min_c=self.params['min_c_s']).to(device)
            # self.concept_critic = conceptLossNet(s_dim, self.n_m_states).to(device)
            self.evolution_model = nextSNet(s_dim, self.n_m_states, self.n_m_actions, n_tasks=self.n_tasks).to(device)
            self.reward_model = rewardNet(s_dim, a_dim, n_tasks=self.n_tasks, lr=self.params['r_lr']).to(device)
            self.R_model = rNet(s_dim, self.n_m_actions, n_tasks=self.n_tasks, lr=self.params['r_lr']).to(device)
            # self.reward_model = RNet(n_m_states, n_tasks=n_tasks, lr=r_lr).to(device)
            # self.meta_critic = np.random.rand(self.n_tasks, self.n_m_states, self.n_m_actions)/self.alpha
            # self.meta_critic1 = np.zeros([self.n_tasks, self.n_m_states, self.n_m_actions])
            # self.meta_critic2 = np.zeros([self.n_tasks, self.n_m_states, self.n_m_actions])
            self.meta_critic1 = np.random.rand(self.n_tasks, self.n_m_states, self.n_m_actions)
            self.meta_critic2 = np.random.rand(self.n_tasks, self.n_m_states, self.n_m_actions)
            if self.multitask:
                self.meta_actor = np.ones([self.n_tasks, self.n_m_states, self.n_m_actions])/self.n_m_actions
                # self.meta_actor = (np.random.rand(self.n_tasks, self.n_m_states, self.n_m_actions)**2)
                # self.meta_actor /= self.meta_actor.sum(2, keepdims=True)
            else:
                self.meta_actor = np.eye(self.n_m_states)[np.newaxis,:,:]

            # self.upper_memory = []
            # for _ in range(0, n_tasks):
            #     self.upper_memory.append(Memory(memory_capacity // (n_tasks*upper_level_period), n_seed=seed))
            self.upper_memory = Memory(self.params['memory_capacity'] // self.upper_level_period, n_seed=self.params['seed'])                  
        
            self.prior_n = 10000
            self.transition_model = torch.ones(self.n_tasks, self.n_m_states, self.n_m_actions, self.n_m_states).to(device)*self.prior_n/self.n_m_states

        updateNet(self.baseline_target, self.baseline, 1.0)
        self.verbose = self.params['verbose']
    
    @property
    def meta_critic(self):
        return np.min([self.meta_critic1.copy(), self.meta_critic2.copy()], 0)
    
    @property
    def PS_T(self):
        return self.concept_model.prior.detach().cpu().numpy()/self.concept_model.prior_n

    def optimize_policy_upper_level(self):
        Q = self.meta_critic.copy()
        PA_ST = self.meta_actor.copy()
        PS_T = self.PS_T.reshape(-1,self.n_m_states,1)
        alpha = self.alpha_upper_level.reshape(-1,self.n_m_states,1)
        
        for _ in range(0,self.upper_policy_steps):
            X = np.log(PA_ST.copy()+1e-6)
            PA = (PA_ST * PS_T).sum(1, keepdims=True).mean(0, keepdims=True)
            X -= self.eta_upper_level / self.n_tasks * PA_ST * PS_T * ( alpha*(X+1.0) + self.beta_upper_level*(np.log(PA+1e-6)+1) - Q )
            X -= X.max(2, keepdims=True)
            expX = np.exp(X)
            Z = expX.sum(2, keepdims=True)
            PA_ST = expX / (Z+1e-10)
        
        self.meta_actor = PA_ST.copy()

    # def meta_learning(self, event):
    #     S = int(event[0])
    #     A = int(event[1])
    #     R = event[2]
    #     nS = int(event[3])
    #     d = event[5]
    #     T = int(event[6])
    #     P_S = (event[7:7+self.n_m_states].copy()).reshape(-1,1)
    #     P_nS = (event[7+self.n_m_states:].copy()).reshape(-1,1)
    def meta_learning(self, S_torch, nS_torch, R_torch, filtered_weights_torch): # a_importance_ratios_torch, task_count_torch, R_torch)
        # # Update value estimation
        # Q_nS = self.meta_critic[T, nS, :]
        # Pi_nS = self.meta_actor[T, nS, :]
        # V_nS = (Pi_nS * (Q_nS - self.alpha_upper_level * np.log(Pi_nS+1e-20))).sum()
        # self.meta_critic[T, S, A] += self.delta_upper_level * (R + self.gamma*(1-d)*V_nS - self.meta_critic[T, S, A])

        # Update value estimation
        # r = np.random.rand()
        # if r > 0.5:
        #     Q_T = self.meta_critic1[T, :, :].copy()            
        # else:
        #     Q_T = self.meta_critic2[T, :, :].copy()

        PA_ST = self.meta_actor.copy()
        PS_T = self.PS_T[:,:,np.newaxis]
        PA = (PA_ST * PS_T).sum(1, keepdims=True).mean(0, keepdims=True)        
        # alpha_T = (self.alpha_upper_level[T,:]).reshape(-1,1)
        
        # V_nS = ((P_A_ST * (Q_T - alpha_T * np.log(P_A_ST + 1e-6) - self.beta_upper_level * np.log(P_A + 1e-6))).sum(1).reshape(-1,1) * P_nS).sum() 
        # if r > 0.5:
        #     self.meta_critic2[T, :, A] += self.delta_upper_level * (R + self.gamma*(1-d)*V_nS - self.meta_critic2[T, :, A].copy()) * P_S.reshape(-1)            
        # else:
        #     self.meta_critic1[T, :, A] += self.delta_upper_level * (R + self.gamma*(1-d)*V_nS - self.meta_critic1[T, :, A].copy()) * P_S.reshape(-1) 

        # Update value estimation
        PS_s = np.zeros([R_torch.size(0),self.n_m_states])
        PnS_ns = np.zeros([R_torch.size(0),self.n_m_states])
        PS_s[np.arange(R_torch.size(0)), S_torch.numpy()] = np.ones(R_torch.size(0))
        PnS_ns[np.arange(R_torch.size(0)), nS_torch.numpy()] = np.ones(R_torch.size(0))
        # PS_s = PS_s_torch.unsqueeze(1).unsqueeze(3).detach().cpu().numpy()
        # PnS_ns = PnS_ns_torch.detach().cpu().numpy()
        filtered_weights = filtered_weights_torch.unsqueeze(2).detach().cpu().numpy()
        # a_importance_ratios = a_importance_ratios_torch.clamp(0.0,10.0).unsqueeze(1).unsqueeze(2).detach().cpu().numpy()
        R = R_torch.view(-1,1).detach().cpu().numpy()

        Q1 = self.meta_critic1.copy()
        Q2 = self.meta_critic2.copy()

        V1 = ((Q2 - self.alpha_upper_level[:,:,np.newaxis] * np.log(PA_ST+1e-10) - self.beta_upper_level * np.log(PA+1e-10)) * PA_ST).sum(2)
        V2 = ((Q1 - self.alpha_upper_level[:,:,np.newaxis] * np.log(PA_ST+1e-10) - self.beta_upper_level * np.log(PA+1e-10)) * PA_ST).sum(2)

        Q1_approx = PS_s[:,np.newaxis,:,np.newaxis] * PA_ST[np.newaxis,:,:,:] * (R + self.gamma * np.einsum('ij,hj->ih', PnS_ns, V1))[:,:,np.newaxis,np.newaxis]
        Q2_approx = PS_s[:,np.newaxis,:,np.newaxis] * PA_ST[np.newaxis,:,:,:] * (R + self.gamma * np.einsum('ij,hj->ih', PnS_ns, V2))[:,:,np.newaxis,np.newaxis]
        
        Q1_error = ((Q1_approx - Q1[np.newaxis,:,:,:]) * filtered_weights).sum(0)
        Q2_error = ((Q2_approx - Q2[np.newaxis,:,:,:]) * filtered_weights).sum(0)
        
        self.meta_critic1 += self.delta_upper_level * Q1_error
        self.meta_critic2 += self.delta_upper_level * Q2_error
        
        # Update policy
        # if (self.bottom_level_epsds + 1) % self.evaluation_upper_level_steps == 0:            
        #     Q = self.meta_critic.copy()
        #     Q = Q - Q.max(2, keepdims=True)
        #     expQ = np.exp(Q/self.alpha_upper_level)
        #     Z = expQ.sum(2, keepdims=True)
        #     self.meta_actor = expQ/Z
        self.optimize_policy_upper_level()
        
        PA_ST = self.meta_actor.copy()
        HA_ST = -(PA_ST * np.log(PA_ST + 1e-10)).sum(2)
        PA = (PA_ST * PS_T).sum(1).mean(0)
        HA = -(PA * np.log(PA + 1e-10)).sum()

        # Update alpha
        # mask = self.alpha_upper_level <= self.alpha_upper_level_threshold
        gradient_step = self.mu_upper_level * self.PS_T / self.n_tasks * (HA_ST - self.threshold_entropy_alpha_upper_level)
        log_alpha = np.log(self.alpha_upper_level+1e-10)
        log_alpha -= gradient_step #* mask
        self.alpha_upper_level = np.exp(log_alpha).clip(1e-10, self.alpha_threshold) #- (1.0-mask) * gradient_step
        
        # Update beta
        if not self.cancel_beta_upper_level:
            # if self.beta_upper_level <= self.beta_upper_level_threshold:
            log_beta = np.log(self.beta_upper_level+1e-10)
            log_beta -= self.mu_upper_level * (HA - self.threshold_entropy_beta_upper_level) 
            self.beta_upper_level = np.exp(log_beta).clip(1e-10, self.alpha_threshold)
            # else:
            #     self.beta_upper_level -= self.mu_upper_level * (HA - self.threshold_entropy_beta_upper_level)
        else:
            self.beta_upper_level = 0.0
        
        # Update delta, eta and mu
        self.delta_upper_level = np.max([self.delta_upper_level*self.rate_delta_upper_level, self.min_delta_upper_level])
        self.eta_upper_level = np.max([self.eta_upper_level*self.rate_delta_upper_level, self.min_eta_upper_level])
        self.mu_upper_level = np.max([self.mu_upper_level*self.rate_delta_upper_level, self.min_mu_upper_level])
    
    def update_transition_model(self, S, A, nS, T, P_nS):
        if self.model_update_method == 'discrete':
            self.transition_model[T,S,A,nS] += 1.0
        else:
            self.transition_model[T,S,A,:] += torch.FloatTensor(P_nS.reshape(-1)).to(device).clone()
        self.transition_model[T,S,A,:] *= self.prior_n/(self.prior_n+1)

    def high_level_decision(self, task, S, explore):
        if (self.bottom_level_epsds % self.upper_level_period) == 0:
            policy = self.meta_actor[task, S, :].reshape(-1)
            if explore:
                A = np.random.multinomial(1, policy/(policy.sum()+1e-10)).argmax()
            else:
                #A = policy.argmax()
                A = np.random.choice(np.flatnonzero(np.isclose(policy, policy.max())))
            self.m_action = A
        else:
            A = self.m_action
        return A

    def concept_inference(self, s, explore=False):
        if (self.bottom_level_epsds % self.upper_level_period) == 0:
            S, PS = self.concept_model.sample_m_state(torch.from_numpy(s).float().to(device), explore=explore)
            # S = self.concept_model.sample_m_state(torch.from_numpy(s).float().to(device), explore=explore)
            self.m_state = S
            self.posterior = PS.detach().cpu().numpy()
        else:
            S = self.m_state
        return S 
    
    def memorize(self, event, init=False):
        if init:
            self.memory.store(event[np.newaxis,:])
        else:
            self.memory.store(event.tolist())
    
    def memorize_in_upper_level(self, event, init=False):
        if init:
            self.upper_memory.store(event[np.newaxis,:])
        else:
            self.upper_memory.store(event.tolist())

    def act(self, s, task, explore=True):
        assert np.all(s==s), 'Invalid state - act'
        if self.hierarchical:        
            S = self.concept_inference(s, explore=explore)
            if self.multitask:
                A = self.high_level_decision(task, S, explore=explore)
            else:
                A = S
        else:
            S, A = 0, 0              
        s_cuda = torch.FloatTensor(s).unsqueeze(0).to(device)
        A_cuda = torch.LongTensor([A]).to(device)
        with torch.no_grad():
            a, llhoods_a = self.actor.sample_action_and_llhood_pairs(s_cuda, A_cuda, explore=explore)[1:]
            assert torch.all(a==a), 'Invalid action - act'
            a, llhoods_a = a.cpu().numpy(), llhoods_a.cpu().numpy()             
            return a, llhoods_a
    
    def update_upper_level(self, r, done, task, completed_episode, state, action, action_llhoods, remember=True, learn=True):
        if remember:
            self.cumulative_reward += r
            self.action_llhoods += action_llhoods.copy()
            self.past_states.append(state.copy())
            self.past_actions.append(action.copy())
            self.time_flow(done, task, completed_episode, r, state, action, action_llhoods, learn=learn)

        self.bottom_level_epsds += 1
        if done or completed_episode:
            self.reset_upper_level()
        

    def time_flow(self, done, task, completed_episode, r, state, action, action_llhoods, learn=True): 
        to_learn_complete = self.stored and (self.bottom_level_epsds % self.upper_level_period) == 0       
        if to_learn_complete:
            if learn and self.hierarchical:
            #     # learning_event = np.empty(7)
            #     learning_event = np.empty(7+2*self.n_m_states)
            #     learning_event[0] = self.past_m_state
            #     learning_event[1] = self.past_m_action
            #     learning_event[2] = self.cumulative_reward - r
            #     learning_event[3] = self.m_state
            #     learning_event[4] = self.m_action
            #     learning_event[5] = float(done)
            #     learning_event[6] = task
            #     learning_event[7:7+self.n_m_states] = self.past_posterior.copy()
            #     learning_event[7+self.n_m_states:] = self.posterior.copy()
            #     self.meta_learning(learning_event.copy())
                # Update prior
                self.concept_model.update_prior(self.past_m_state, task, self.past_posterior.copy(), self.model_update_method)
                self.update_transition_model(self.past_m_state, self.past_m_action, self.m_state, task, self.posterior.copy())

            storing_event = np.empty(self.N_s_dim+self.upper_level_period*self.a_dim+self.n_m_actions+3)
            for i in range(0, self.upper_level_period+1):
                storing_event[i*self.s_dim:(i+1)*self.s_dim] = self.past_states[i].copy()
                if i < self.upper_level_period:
                    storing_event[self.N_s_dim + i*self.a_dim : self.N_s_dim + (i+1)*self.a_dim] = self.past_actions[i].copy()
            storing_event[self.N_s_dim + self.upper_level_period*self.a_dim:-3] = self.action_llhoods - action_llhoods.copy()
            storing_event[-3] = self.past_m_action
            storing_event[-2] = task
            storing_event[-1] = self.cumulative_reward - r
            self.memorize_in_upper_level(storing_event)
            # self.train_reward_model(task, self.past_states, self.cumulative_reward - r)
            # self.train_evolution_model(task, self.past_states, state, self.past_m_action)

            self.past_states = [state.copy()]
            self.past_actions = [action.copy()]
            self.action_llhoods = action_llhoods.copy()
            self.cumulative_reward = r
            self.stored = False
            # self.alpha_upper_level = np.max([0.999993*self.alpha_upper_level, 5e-2])

        if not self.stored and (self.bottom_level_epsds % self.upper_level_period) == 0:
            self.past_m_state = self.m_state
            self.past_m_action = self.m_action
            self.past_posterior = self.posterior.copy()
            # self.past_states = state.copy()
            self.stored = True
        
        if done and not to_learn_complete and self.hierarchical and learn:
        #     # event = np.empty(7)
        #     event = np.empty(7+2*self.n_m_states)
        #     event[0] = self.m_state
        #     event[1] = self.m_action
        #     event[2] = self.cumulative_reward
        #     event[3] = self.m_state
        #     event[4] = self.m_action
        #     event[5] = float(done)
        #     event[6] = task
        #     event[7:7+self.n_m_states] = self.past_posterior.copy()
        #     PS = self.concept_model.sample_m_state(torch.from_numpy(state).float().to(device), explore=True)[1]
        #     event[7+self.n_m_states:] = PS.detach().cpu().numpy().copy()
        #     self.meta_learning(event.copy())  
            # Update prior
            self.concept_model.update_prior(self.m_state, task, self.past_posterior.copy(), self.model_update_method)
            PS = self.concept_model.sample_m_state(torch.from_numpy(state).float().to(device), explore=True)[1]
            self.update_transition_model(self.m_state, self.m_action, self.m_state, task, PS.detach().cpu().numpy().copy())          

    def reset_upper_level(self):
        self.past_states = []
        self.past_actions = []
        self.action_llhoods = np.zeros((1, self.n_m_actions))
        self.cumulative_reward = 0.0            
        self.bottom_level_epsds = 0
        self.stored = False    

    def learn_skills(self, only_metrics=False):
        batch = self.memory.sample(self.policy_batch_size)
        batch = np.array(batch)

        if batch.shape[0] > 0:
            s_batch = torch.FloatTensor(batch[:,:self.s_dim]).to(device)
            a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
            r_batch = torch.FloatTensor(batch[:,self.sa_dim]).unsqueeze(1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.sa_dim+1:self.sars_dim]).to(device)
            d_batch = torch.FloatTensor(batch[:,self.sars_dim]).unsqueeze(1).to(device)
            T_batch = batch[:,self.sarsd_dim+1].astype('int')        

            if not only_metrics:
                # Optimize q networks
                q1 = self.critic1(s_batch, a_batch)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                q2 = self.critic2(s_batch, a_batch)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                next_v = self.baseline_target(ns_batch)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                q_approx = r_batch + self.gamma * next_v * (1-d_batch)
                assert q_approx.size(1) == 1, 'Wrong size'

                q1_loss = self.critic1.loss_func(q1, q_approx.detach())
                self.critic1.optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1.parameters(), self.clip_value)
                self.critic1.optimizer.step()
                
                q2_loss = self.critic2.loss_func(q2, q_approx.detach())
                self.critic2.optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2.parameters(), self.clip_value)
                self.critic2.optimizer.step()

                # Optimize reward model
                r_batch_off = self.reward_model(s_batch, a_batch)[np.arange(T_batch.shape[0]), T_batch]
                r_loss = self.reward_model.loss_func(r_batch_off, r_batch.detach())
                self.reward_model.optimizer.zero_grad()
                r_loss.backward()
                clip_grad_norm_(self.reward_model.parameters(), self.clip_value)
                self.reward_model.optimizer.step()

            # Optimize v network
            if self.hierarchical:
                PS_s = self.concept_model.sample_m_state_and_posterior(s_batch)[1]
                PA_ST = torch.from_numpy(self.meta_actor).float().to(device)
                PA_sT = torch.einsum('hjk,ij->ikh', PA_ST, PS_s.detach())
                HA_sT = -(PA_sT * torch.log(PA_sT + 1e-12)).sum(1)

                a_batch_A, log_Pa_sAp_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch)
                a_batch_A = a_batch_A.view(-1, self.a_dim)
                s_batch_repeated = s_batch.view(s_batch.size(0),1,self.s_dim).repeat(1,self.n_m_actions,1).view(-1,self.s_dim)

                q1_saT_A = self.critic1(s_batch_repeated.detach(), a_batch_A)
                q2_saT_A = self.critic2(s_batch_repeated.detach(), a_batch_A)
                q_saT_A = torch.min(torch.stack([q1_saT_A, q2_saT_A]), 0)[0]
                q_saT_A = q_saT_A.view(-1, self.n_m_actions, self.n_tasks)
                q_saT = torch.einsum('ikh,ikh->ih', PA_sT, q_saT_A)

                log_Pa_sA = log_Pa_sAp_A[:, np.arange(self.n_m_actions), np.arange(self.n_m_actions)]
                Ha_A_sT = -torch.einsum('ikh,ik->ih', PA_sT, log_Pa_sA)

                # log_Pa_sT_A = torch.logsumexp(log_Pa_sAp_A.unsqueeze(3) + torch.log(PA_sT + 1e-12).unsqueeze(2), dim=1)
                # log_PAp_saT_A = log_Pa_sAp_A.unsqueeze(3) + torch.log(PA_sT + 1e-12).unsqueeze(2) - log_Pa_sT_A.unsqueeze(1)
                # HAp_saT_A = -(torch.exp(log_PAp_saT_A) * log_PAp_saT_A).sum(1)
                # HA_a_sT = torch.einsum('ikh,ikh->ih', PA_sT, HAp_saT_A)
                log_Pa_sT_A = torch.logsumexp(log_Pa_sAp_A.unsqueeze(3) + torch.log(PA_sT + 1e-12).unsqueeze(2), dim=1)
                log_PA_saT = log_Pa_sA.unsqueeze(2) + torch.log(PA_sT + 1e-12) - log_Pa_sT_A
                HA_a_sT = -torch.einsum('ikh,ikh->ih', PA_sT, log_PA_saT)
                Ha_sT = -torch.einsum('ikh,ikh->ih', PA_sT, log_Pa_sT_A)

                v_approx = q_saT + self.mu.view(1,1,-1) * Ha_A_sT - (self.alpha.view(1,1,-1) + self.mu.view(1,1,-1)) * HA_a_sT 

                if not only_metrics:
                    v = self.baseline(s_batch)

                alpha_gradient = (Ha_sT.detach() - Ha_A_sT.detach() - (HA_sT.detach() - self.threshold_entropy_alpha).clamp(0.0,np.log(self.n_m_actions))).mean(0)
                mu_gradient = (Ha_sT.detach() - self.threshold_entropy_mu).mean(0)
                    
                # if self.multitask:  
                #     policy = torch.from_numpy(self.meta_actor)[T_batch,:,:].float().to(device)
                #     policy_off = self.meta_actor[T_batch.astype(int),S_batch_off.numpy(),:]
                #     assert len(policy_off.shape) == 2, 'Wrong size'
                #     assert policy_off.shape[0] == S_batch_off.size(0), 'Wrong size'
                #     assert policy_off.shape[1] == self.n_m_actions, 'Wrong size'
                #     A_batch_off = torch.from_numpy(vectorized_multinomial(policy_off)).long().to(device)                           
                #     _, a_batch_off, a_conditional_llhood_off = self.actor.sample_action_and_llhood_pairs(s_batch[:,:self.s_dim], A_batch_off)
                #     A_probability_off =  torch.einsum('ijk,ij->ik', policy, PS_s)                                    
                # else:
                #     A_batch_off = S_batch_off.clone()
                #     _, a_batch_off, a_conditional_llhood_off = self.actor.sample_action_and_llhood_pairs(s_batch[:,:self.s_dim], A_batch_off)
                #     A_probability_off = PS_s.detach() 
                # a_log_posterior_unnormalized = a_conditional_llhood_off + torch.log(A_probability_off + 1e-12)
                # a_llhood_off = torch.logsumexp(a_log_posterior_unnormalized+1e-12, dim=1, keepdim=True)
            else:
                A_batch_off = torch.zeros(s_batch.size(0),).long().to(device)
                PA_sT = torch.ones(s_batch.size(0),).long().to(device)
                _, a_batch_off, a_conditional_llhood_off = self.actor.sample_action_and_llhood_pairs(s_batch[:,:self.s_dim], A_batch_off)
                log_Pa_sT_A =  a_conditional_llhood_off
                H_a_sA = -a_conditional_llhood_off[np.arange(s_batch.size(0)), A_batch_off].view(-1,1)

                if not only_metrics:
                    q1_off = self.critic1(s_batch.detach(), a_batch_off)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                    q2_off = self.critic2(s_batch.detach(), a_batch_off)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                    q_off = torch.min(q1_off, q2_off)
                    assert q_off.size(1) == 1, 'Wrong size'
                
                    v = self.baseline(s_batch)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                    assert v.size(1) == 1, 'Wrong size'
                    v_approx = q_off - self.alpha*log_Pa_sT_A
                    assert log_Pa_sT_A.size(1) == 1, 'Wrong size'
                    assert v_approx.size(1) == 1, 'Wrong size'   
                
            if not only_metrics:
                v_loss = ((v - v_approx.detach())**2).sum(1).mean()
                self.baseline.optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.baseline.parameters(), self.clip_value)
                self.baseline.optimizer.step()
                updateNet(self.baseline_target, self.baseline, self.soft_lr)

                # Optimize skill network
                # pi_loss = (-q_saT - self.alpha.view(1,1,-1) * Ha_A_sT).mean()
                pi_loss = (-v_approx).mean()
                self.actor.optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                # pi_loss.backward()
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()

                # pi_loss_mean = (self.mu.view(1,1,-1) * HA_a_sT).mean()
                # self.actor.optimizer_mean.zero_grad()
                # pi_loss_mean.backward()
                # clip_grad_norm_(self.actor.parameters_mean, self.clip_value)
                # self.actor.optimizer_mean.step()

                # Optimize dual variables
                if self.automatic_lower_temperature:
                    # Optimize mu
                    # mu_mask = (self.mu < self.mu_threshold).float()
                    log_mu = torch.log(self.mu + 1e-6)
                    log_mu -= self.tau_mu * mu_gradient #* mu_mask
                    self.mu = torch.exp(log_mu).clamp(1e-10, self.mu_threshold) #+ (1-mu_mask) * self.tau_mu * mu_gradient
                    # Optimize alpha
                    # alpha_mask = (self.alpha < self.alpha_threshold).float()
                    log_alpha = torch.log(self.alpha + 1e-6)
                    log_alpha -= self.tau_alpha * alpha_gradient #* alpha_mask
                    self.alpha = torch.exp(log_alpha).clamp(1e-10, self.alpha_threshold) #- (1-alpha_mask) * self.tau_alpha * alpha_gradient


        else:
            log_Pa_sT_A = torch.zeros(1).to(device)  
            Ha_A_sT = torch.zeros(1).to(device)
            PA_sT = torch.ones(1,1).to(device)/self.n_m_actions
            PA_ST = torch.ones(1,1).to(device)/self.n_m_actions
            HA_a_sT = torch.zeros(1).to(device)
            HA_sT = torch.zeros(1).to(device)            

        if only_metrics:
            marginal_A = PA_sT.mean(0) 
            PS_T = self.concept_model.prior.detach().unsqueeze(2)/self.concept_model.prior_n
            metrics = {
                'H(a|A,s)': Ha_A_sT.mean().detach().cpu().numpy(),
                'H(A)': -(marginal_A * torch.log(marginal_A + 1e-12)).sum(0).mean().detach().cpu().numpy(),
                'H(A|a,s)': HA_a_sT.mean().detach().cpu().numpy(),
                'H(A|S)': -(PA_ST * PS_T * torch.log(PA_ST + 1e-12)).sum((1,2)).mean().detach().cpu().numpy(),
                'H(A|s)': HA_sT.mean().detach().cpu().numpy()
            }
            
            return metrics
    
    # def learn_reward(self, batch=[]):
        # if self.hierarchical and not self.upper_memory.empty():
        #     upper_batch = self.upper_memory.sample(self.concept_batch_size)
        #     upper_batch = np.array(upper_batch)
            
        #     # a_llhoods_upper_batch = torch.FloatTensor(upper_batch[:,self.N_s_dim + self.upper_level_period*self.a_dim:-3]).to(device)
        #     T_upper_batch = upper_batch[:,-2].astype('int')
        #     R_upper_batch = torch.FloatTensor(upper_batch[:,-1]).unsqueeze(1).to(device)
        #     s_upper_batch = torch.FloatTensor(upper_batch[:, 0: self.s_dim]).to(device)

        #     # a_llhoods_upper_batch_off = torch.zeros_like(a_llhoods_upper_batch)
        #     # for i in range(0, self.upper_level_period+1):
        #     #     s_upper_batch_ = torch.FloatTensor(upper_batch[:, i*self.s_dim:(i+1)*self.s_dim]).to(device)
        #     #     if i < self.upper_level_period:
        #     #         a_upper_batch = torch.FloatTensor(upper_batch[:, self.N_s_dim + i*self.a_dim : self.N_s_dim + (i+1)*self.a_dim]).to(device)
        #     #         a_llhoods_upper_batch_off += self.actor.llhoods(s_upper_batch_, a_upper_batch).detach()

        #     # Optimize reward network
        #     # a_importance_ratios = torch.exp((a_llhoods_upper_batch_off - a_llhoods_upper_batch).clamp(-1e6,10))+1e-10
        #     R_llhood = self.reward_model.llhood(R_upper_batch, s_upper_batch, T_upper_batch)
        #     R_model_loss = -(R_llhood).mean() # * a_importance_ratios.detach()).mean()
        #     self.reward_model.optimizer.zero_grad()
        #     R_model_loss.backward()
        #     clip_grad_norm_(self.reward_model.parameters(), self.clip_value)
        #     self.reward_model.optimizer.step()
        # if not self.memory.empty():
        #     if len(batch) == 0:
        #         batch = self.upper_memory.sample(self.concept_batch_size)
        #         batch = np.array(upper_batch)
                    
        #     s_batch = torch.FloatTensor(batch[:,0:self.s_dim]).to(device)
        #     a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
        #     T_batch = batch[:,-2].astype('int')
        #     R_upper_batch = torch.FloatTensor(upper_batch[:,-1]).unsqueeze(1).to(device)
        
        #     R_batch_off = self.reward_model(s_upper_batch, a_upper_batches)[np.range(T_upper_batch.shape[0]), T_upper_batch]
        #     R_loss = self.reward_model.loss_func(R_batch_off, R_upper_batch.detach())
        #     self.reward_model.optimizer.zero_grad()
        #     R_loss.backward()
        #     clip_grad_norm_(self.reward_model.parameters(), self.clip_value)
        #     self.reward_model.optimizer.step()
    def learn_reward(self, upper_batch=[]):
        if not self.upper_memory.empty():
            if len(upper_batch) == 0: 
                upper_batch = self.upper_memory.sample(self.concept_batch_size) 
                upper_batch = np.array(upper_batch)
        
            batch_size = upper_batch.shape[0]
            T_upper_batch = upper_batch[:,-2].astype('int')
            
            s_upper_batch = torch.FloatTensor(upper_batch[:,0:self.s_dim]).to(device)
            a_upper_batch_off = self.actor.sample_actions(s_upper_batch).view(-1,self.a_dim)
            s_batch_repeated = s_upper_batch.view(batch_size,1,self.s_dim).repeat(1,self.n_m_actions,1).view(-1,self.s_dim)

            R_batch_off = self.reward_model(s_batch_repeated, a_upper_batch_off).view(-1,self.n_m_actions,self.n_tasks)[np.arange(batch_size),:,T_upper_batch].detach()            
            R_llhood = self.R_model.llhood(R_batch_off, s_upper_batch, T_upper_batch)

            R_model_loss = -(R_llhood).mean() 
            self.R_model.optimizer.zero_grad()
            R_model_loss.backward()
            clip_grad_norm_(self.R_model.parameters(), self.clip_value)
            self.R_model.optimizer.step()

    def learn_concepts(self, only_metrics=False, upper_batch=[]):                                       
        if not self.upper_memory.empty():
            if len(upper_batch) == 0: 
                upper_batch = self.upper_memory.sample(self.concept_batch_size) 
                upper_batch = np.array(upper_batch)
            
            batch_size = upper_batch.shape[0]
            baseline = np.log(batch_size)

            T_upper_batch = upper_batch[:,-2].astype('int')
            R_upper_batch = torch.FloatTensor(upper_batch[:,-1]).unsqueeze(1).to(device)
            
            s_upper_batches = []
            a_upper_batches_off = []
            log_Ptrajectory_A = torch.zeros(batch_size, self.n_m_actions).to(device)
            for i in range(0, self.upper_level_period+1):
                s_upper_batches.append(torch.FloatTensor(upper_batch[:, i*self.s_dim:(i+1)*self.s_dim]).to(device))
                if i < self.upper_level_period:
                    a_upper_batch = torch.FloatTensor(upper_batch[:, self.N_s_dim + i*self.a_dim : self.N_s_dim + (i+1)*self.a_dim]).to(device)
                    log_Ptrajectory_A += self.actor.llhoods(s_upper_batches[-1], a_upper_batch).detach()
                    a_upper_batches_off.append(self.actor.sample_actions(s_upper_batches[-1]))
                    
            # a_upper_batches_off = torch.cat(a_upper_batches_off, 2).view(-1,self.a_dim)
            a_upper_batch_off = a_upper_batches_off[0].view(-1,self.a_dim)
            s_batch_repeated = s_upper_batches[0].view(batch_size,1,self.s_dim).repeat(1,self.n_m_actions,1).view(-1,self.s_dim)

            R_batch_off = self.reward_model(s_batch_repeated, a_upper_batch_off).view(-1,self.n_m_actions,self.n_tasks)[np.arange(batch_size),:,T_upper_batch].detach()
            # if not only_metrics:
            #     R_llhood = self.R_model.llhood(R_batch_off, s_upper_batches[0], T_upper_batch)
            #     R_model_loss = -(R_llhood).mean() 
            #     self.R_model.optimizer.zero_grad()
            #     R_model_loss.backward()
            #     clip_grad_norm_(self.R_model.parameters(), self.clip_value)
            #     self.R_model.optimizer.step()
            # upper_batch_2 = self.upper_memory.sample(self.concept_batch_size)
            # s2_upper_batch = torch.FloatTensor(np.array(upper_batch_2)[:, 0: self.s_dim]).to(device)

            # del upper_batch
            # del upper_batch_2
            
            # I(s_t:S_t)
            S, PS_s, log_PS_s = self.concept_model.sample_m_state_and_posterior(s_upper_batches[0])
            PS = PS_s.mean(0).view(-1,1)
            log_PS = -baseline + torch.logsumexp(log_PS_s, dim=0).view(-1,1)
            HS = -(PS * log_PS).sum()
            HS_s = -(PS_s * log_PS_s).sum(1).mean()
            s_S_mutual_information = HS - HS_s

            # PS_T = self.concept_model.prior.detach()/self.concept_model.prior_n
            # PA_ST = torch.from_numpy(self.meta_actor).float().to(device)
            # log_PAtrajectory_sST = a_llhoods_upper_batch_off.unsqueeze(1).unsqueeze(2) + torch.log(PA_ST + 1e-12).unsqueeze(0)
            # log_Ptrajectory_sST = torch.logsumexp(log_PAtrajectory_sST + 1e-10, dim=3, keepdim=True)
            # PA_trajectorysST = torch.exp(log_PAtrajectory_sST - log_Ptrajectory_sST)
            # PA_trajectorysST /= PA_trajectorysST.sum(3, keepdim=True)
            # PA_trajectorysT = (PA_trajectorysST * PS_T.unsqueeze(0).unsqueeze(3)).sum(2)[np.arange(batch_size),T_upper_batch,:]
            # a_importance_ratios = torch.exp((a_llhoods_upper_batch_off - a_llhoods_upper_batch).clamp(-1e6,10))+1e-10
            # if not only_metrics:
            #     # Optimize reward network
            #     R_llhood = self.reward_model.llhood(R_upper_batch, s_upper_batches[0], T_upper_batch)
            #     R_model_loss = -(R_llhood * PA_trajectorysT.detach()).mean() 
            #     self.reward_model.optimizer.zero_grad()
            #     R_model_loss.backward()
            #     clip_grad_norm_(self.reward_model.parameters(), self.clip_value)
            #     self.reward_model.optimizer.step()

            # I(R_t:S_t|T_t)
            # Pr_ssAA_T = torch.exp(self.reward_model.sample_and_cross_llhood(s_upper_batches[0], T_upper_batch))
            # assert len(Pr_ssAA_T.shape) == 4, 'P(R|s,A,T) is calculated incorrectly. Wrong size.'
            # assert Pr_ssAA_T.size(0) == batch_size, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 0.'
            # assert Pr_ssAA_T.size(1) == batch_size, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 1.'
            # assert Pr_ssAA_T.size(2) == self.n_m_actions, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 2.'
            # assert Pr_ssAA_T.size(3) == self.n_m_actions, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 2.'
            
            # policy_upper = self.meta_actor[T_upper_batch, :, :]
            # policy_upper = torch.from_numpy(policy_upper).float().to(device)
            # Pr_ssAS_T = torch.einsum('ikp,ijtp->ijkt', policy_upper, Pr_ssAA_T)
            # cross_task_mask = (T_upper_batch.reshape(-1,1) == T_upper_batch.reshape(1,-1))*1.0
            # cross_task_mask /= (cross_task_mask.sum(1, keepdims=True) + 1e-10)
            # cross_task_mask = torch.from_numpy(cross_task_mask).float().to(device)
            # Pr_ssAS_T_valid = Pr_ssAS_T * cross_task_mask.unsqueeze(2).unsqueeze(3)            
            # Pr_sSA_T_unnormalized = torch.einsum('ijkt,jk->ikt', Pr_ssAS_T_valid, PS_s)

            # task_mask = torch.zeros(batch_size, self.n_tasks).float().to(device)
            # task_mask[np.arange(batch_size), T_upper_batch] = torch.ones(batch_size).float().to(device)
            # task_count = task_mask.sum(0).view(-1,1)  
            # PS_s_classified_in_task = torch.zeros(batch_size, self.n_tasks, self.n_m_states).float().to(device)
            # PS_s_classified_in_task[np.arange(batch_size), T_upper_batch, :] = PS_s
            # PS_T = PS_s_classified_in_task.sum(0) / (task_count + 1e-10)

            # # S_probability_given_T = self.concept_model.prior[T_upper_batch, :]/self.concept_model.prior_n            
            # # Pr_sSA_T = Pr_sSA_T_unnormalized / (S_probability_given_T.unsqueeze(2) + 1e-10)
            # Pr_sSA_T = Pr_sSA_T_unnormalized / (PS_T[T_upper_batch, :].unsqueeze(2) + 1e-10)
            # Pr_sA_T = Pr_sSA_T_unnormalized.sum(1, keepdim=True)
            
            # HR_ST = -(policy_upper * PS_s.unsqueeze(2) * torch.log(Pr_sSA_T + 1e-10)).sum((1,2)).view(-1,1)
            # HR_T = -(policy_upper * PS_s.unsqueeze(2) * torch.log(Pr_sA_T + 1e-10)).sum((1,2)).view(-1,1)  
            # R_S_mutual_information = HR_T - HR_ST   
             
            # I(R_t:S_t|T_t)
            log_PR_spApT_sA = self.R_model.llhood(R_batch_off, s_upper_batches[0], T_upper_batch, cross=True)
            PA_ST = torch.from_numpy(self.meta_actor).float().to(device)
            PSA_sT = PS_s.unsqueeze(2) * PA_ST[T_upper_batch,:,:]
            log_PSR_spT_sA = torch.logsumexp(torch.log(PSA_sT + 1e-10).unsqueeze(1).unsqueeze(4) + log_PR_spApT_sA.unsqueeze(2), dim=3)

            task_mask = torch.zeros(batch_size, self.n_tasks).float().to(device)
            task_mask[np.arange(batch_size), T_upper_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask /= (task_count.view(1,-1) + 1e-10)
            PS_s_classified_in_task = (1e-10)*torch.ones(batch_size, self.n_tasks, self.n_m_states).float().to(device)
            PS_s_classified_in_task[np.arange(batch_size), T_upper_batch, :] = PS_s
            PS_T = PS_s_classified_in_task.sum(0) / (task_count + 1e-10)
            PS_T /= PS_T.sum(1, keepdim=True).detach()
            
            cross_task_mask = (T_upper_batch.reshape(-1,1) == T_upper_batch.reshape(1,-1))*1.0
            cross_task_mask /= (cross_task_mask.sum(1, keepdims=True) + 1e-10)
            cross_task_mask = torch.from_numpy(cross_task_mask).float().to(device)
            log_PR_ST_sA_unnormalized = torch.logsumexp(torch.log(cross_task_mask + 1e-10).unsqueeze(2).unsqueeze(3) + log_PSR_spT_sA, dim=0)
            log_PR_ST_sA = log_PR_ST_sA_unnormalized - torch.log(PS_T[T_upper_batch, :] + 1e-10).unsqueeze(2)
            log_PR_T_sA = torch.logsumexp(log_PR_ST_sA_unnormalized, dim=1)

            PA_s_T = PSA_sT.sum(1)
            HR_T = -(PA_s_T * log_PR_T_sA).sum(1).mean()
            HR_ST = -(PSA_sT * log_PR_ST_sA).sum((1,2)).mean()  
            R_S_mutual_information = HR_T - HR_ST    
            
            # I(S_t+1:S_t|A_t,T_t)
            # assert torch.all(a_importance_ratios != float('inf')), 'Infinite lklhoods'
            trajectory_weights_A = torch.exp(log_Ptrajectory_A - torch.logsumexp(log_Ptrajectory_A + 1e-10, dim=0, keepdim=True))
            trajectory_weights_A /= trajectory_weights_A.sum(0, keepdim=True)
            filtered_weights = (trajectory_weights_A.unsqueeze(1) * task_mask.unsqueeze(2)).detach()
            filtered_weights /= (filtered_weights.sum(0, keepdim=True) + 1e-10)
            nS, PnS_ns = self.concept_model.sample_m_state_and_posterior(s_upper_batches[-1])[0:2]
            PnS_SAT_unnormalized = ((filtered_weights.unsqueeze(2).unsqueeze(4) * PS_s.unsqueeze(1).unsqueeze(3).unsqueeze(4) * PnS_ns.unsqueeze(1).unsqueeze(2).unsqueeze(3))).sum(0)
            PnS_SAT = PnS_SAT_unnormalized / (PS_T.unsqueeze(2).unsqueeze(3) + 1e-10)

            PA_T = (PA_ST * PS_T.unsqueeze(2)).sum(1, keepdim=True)
            PnS_AT = torch.einsum('hjkl,hjk->hkl', PnS_SAT_unnormalized, PA_ST/(PA_T + 1e-10))

            PTSAnS = (PS_T.unsqueeze(2) * PA_ST).unsqueeze(3) * PnS_SAT * task_count.view(-1,1,1,1) / batch_size
            PTAnS = PTSAnS.sum(1)
            HnS_SAT = -(PTSAnS * torch.log(PnS_SAT + 1e-10)).sum()
            HnS_AT = -(PTAnS * torch.log(PnS_AT + 1e-10)).sum()
            nS_S_mutual_information = HnS_AT - HnS_SAT

            # PS = (self.concept_model.prior.detach()/self.concept_model.prior_n).mean(0).view(-1,1)
            # PA_S = PA_ST.mean(0)
            # PAS = PA_S * PS
            # PA = PAS.sum(0)    

            # assert torch.all(PS==PS), 'Problems with prior'
            # assert torch.all(PA_S==PA_S), 'Problems with PA_S'
            # assert torch.all(PAS==PAS), 'Problems with PAS'
            # assert torch.all(PA==PA), 'Problems with PA'

            # PnS_SA = PnS_SA_unnormalized / (PS.unsqueeze(2) + 1e-10)
            # PnS_SA /= (PnS_SA.sum(2, keepdim=True) + 1e-10)
            # PnSSA = PnS_SA * PAS.unsqueeze(2) 
            # PnS_A = PnSSA.sum(0) / (PA.unsqueeze(1) + 1e-10)

            # assert torch.all(PnS_SA==PnS_SA), 'Problems with model'
            # assert torch.all(PnS_A==PnS_A), 'Problems with model 2'
            
            # HnS_SA = -(PnSSA * torch.log(PnS_SA + 1e-10)).sum()
            # HnS_A = -(PnSSA.sum(0) * torch.log(PnS_A + 1e-10)).sum()
            # nS_S_mutual_information = HnS_A - HnS_SA

            PA_sT = torch.einsum('hjk,ij->ikh', PA_ST, PS_s)
            HA_sT = -(PA_sT * torch.log(PA_sT + 1e-12)).sum(1)

            disentanglement_metric = self.beta*s_S_mutual_information + self.eta*R_S_mutual_information + self.nu*nS_S_mutual_information

            if not only_metrics:
                self.meta_learning(S, nS, R_upper_batch, filtered_weights)                       

            # Inconsitency metric
            S_upper_posteriors = [(PS_s, log_PS_s)]
            posterior_inconsistency = torch.zeros_like(PS_s)
            for s_batch in s_upper_batches[1:]:
                S_upper_posteriors.append(self.concept_model.sample_m_state_and_posterior(s_batch)[1:])
                if self.inconsistency_metric == 'poly':
                    posterior_inconsistency += (2*(S_upper_posteriors[-1][0] - S_upper_posteriors[-2][0].detach()))**4
                else:
                    posterior_inconsistency += S_upper_posteriors[-2][0].detach()*(S_upper_posteriors[-2][1].detach() - S_upper_posteriors[-1][1] + 10.0*S_upper_posteriors[-1][0])

            if self.inconsistency_metric == 'poly':
                inconsistency_metric = self.zeta * posterior_inconsistency.mean(1, keepdim=True)
            else:
                inconsistency_metric = self.zeta * posterior_inconsistency.sum(1, keepdim=True)

            # Action divergence
            point_distributions = self.estimate_macro_policies(s_upper_batches[0])  
            KL_divergence_Pis_Pi_ST = (point_distributions.transpose(1,2).unsqueeze(2) * (torch.log(point_distributions.transpose(1,2).unsqueeze(2) + 1e-10) - torch.log(PA_ST.unsqueeze(0) + 1e-10))).sum(3)
            KL_divergence_Pi_Pis_ST = (PA_ST.unsqueeze(0) * (torch.log(PA_ST.unsqueeze(0) + 1e-10) - torch.log(point_distributions.transpose(1,2).unsqueeze(2) + 1e-10))).sum(3)
            JS_divergence_Pi_Pis_ST = 0.5 * (KL_divergence_Pis_Pi_ST + KL_divergence_Pi_Pis_ST)
            JS_divergence_Pi_Pis_S = torch.einsum('ihj,hj->ij', JS_divergence_Pi_Pis_ST, PS_T/self.n_tasks)
            JS_divergence_Pi_Pis = torch.einsum('ij,ij->i', JS_divergence_Pi_Pis_S, PS_s/(PS.view(1,-1)+1e-10))
            action_divergence_metric = self.xi * JS_divergence_Pi_Pis.view(-1,1)
            # action_KL_divergence = (point_distributions.unsqueeze(1) * (torch.log(point_distributions.unsqueeze(1) + 1e-12) - torch.log(point_distributions.unsqueeze(0) + 1e-12))).sum(2)
            # P_S_T = self.concept_model.prior.transpose(0,1)/self.concept_model.prior_n
            # macro_state_correlation = ((PS_s.unsqueeze(2) * PS_T.transpose(0,1).unsqueeze(0)).unsqueeze(1) * PS_s.unsqueeze(0).unsqueeze(3) / (PS.view(1,1,-1,1) + 1e-10)**2).sum(2)
            # action_divergence_metric = self.xi * (0.5 * macro_state_correlation * action_KL_divergence.detach()).mean(2).mean(1, keepdim=True)

            # Optimize concept networks
            if not only_metrics:   
                assert torch.all(action_divergence_metric==action_divergence_metric), 'Problems with D1'
                assert torch.all(inconsistency_metric==inconsistency_metric), 'Problems D2'  
                assert torch.all(disentanglement_metric==disentanglement_metric), 'Problems D3'                          
                concept_model_loss = (action_divergence_metric + inconsistency_metric - disentanglement_metric + 0.01*((self.alpha.view(1,-1) + self.mu.view(1,-1)) * HA_sT).mean(1, keepdim=True)).mean()
                self.concept_model.optimizer.zero_grad()
                concept_model_loss.backward()
                clip_grad_norm_(self.concept_model.parameters(), self.clip_value)
                self.concept_model.optimizer.step()
                assert torch.all(self.concept_model.l1.weight==self.concept_model.l1.weight), 'Invalid concept model parameters'

        else:
            PS_s = torch.zeros(1,1).to(device)
            HS = torch.zeros(1).to(device)
            HS_s = torch.zeros(1).to(device)
            s_S_mutual_information = torch.zeros(1).to(device)
            HnS_AT = torch.zeros(1).to(device)
            HnS_SAT = torch.zeros(1).to(device)                    
            nS_S_mutual_information = torch.zeros(1).to(device)
            HR_ST = torch.zeros(1).to(device)
            HR_T = torch.zeros(1).to(device)
            R_S_mutual_information = torch.zeros(1).to(device)
            disentanglement_metric = torch.zeros(1).to(device)  
            inconsistency_metric = torch.zeros(1).to(device)
            action_divergence_metric = torch.zeros(1).to(device)          

        if only_metrics:
            metrics = {
                'n concepts': len(np.unique(PS_s.argmax(1).detach().cpu().numpy())),
                'H(S)': HS.mean().detach().cpu().numpy(),
                'H(S|s)': HS_s.mean().detach().cpu().numpy(),
                'I(S:s)': s_S_mutual_information.mean().detach().cpu().numpy(), 
                'H(R|S,T)': HR_ST.mean().detach().cpu().numpy(),
                'H(R|T)': HR_T.mean().detach().cpu().numpy(),
                'I(R:S|T)': R_S_mutual_information.mean().detach().cpu().numpy(),
                'H(nS|A,T)': HnS_AT.mean().detach().cpu().numpy(),
                'H(nS|S,A,T)': HnS_SAT.mean().detach().cpu().numpy(),
                'I(nS:S|A,T)': nS_S_mutual_information.mean().detach().cpu().numpy(),
                'D1': disentanglement_metric.mean().detach().cpu().numpy(),
                'D2': inconsistency_metric.mean().detach().cpu().numpy(),
                'D3': action_divergence_metric.mean().detach().cpu().numpy()
            }
            
            return metrics

    def estimate_macro_policies(self, s):
        with torch.no_grad():
            a = self.actor.calculate_mean(s).view(-1, self.a_dim)
            s_repeated = s.view(s.size(0),1,s.size(1)).repeat(1,self.n_m_actions,1).view(-1,self.s_dim)
            
            q = (0.5*(self.critic1(s_repeated, a) + self.critic2(s_repeated, a))).view(-1, self.n_m_actions, self.n_tasks)
            exp_q = torch.exp(q - q.max(1, keepdim=True)[0])
            Pi = exp_q / (exp_q.sum(1, keepdim=True) + 1e-10)
            
        return Pi

    def estimate_metrics(self):
        with torch.no_grad():
            skill_metrics = self.learn_skills(only_metrics=True)
            concept_metrics = self.learn_concepts(only_metrics=True)
        metrics = {**skill_metrics, **concept_metrics}
        return metrics

    def learn_lower(self):
        if self.parallel_learning or not self.hierarchical:
            self.learn_skills()
            if self.hierarchical:
                self.learn_concepts()            
        else:
            if self.bottom_level_epsds % (self.skill_steps + self.concept_steps) < self.skill_steps:
                if self.reward_learning == 'always':
                    self.learn_reward()
                self.learn_skills()
            else:
                if not self.upper_memory.empty():
                    upper_batch = self.upper_memory.sample(self.concept_batch_size) 
                    upper_batch = np.array(upper_batch)
                    self.learn_reward(upper_batch=upper_batch)
                    self.learn_concepts(upper_batch=upper_batch)
    
    def save(self, common_path, specific_path):
        self.params['alpha_upper_level'] = self.alpha_upper_level
        self.params['beta_upper_level'] = self.beta_upper_level
        pickle.dump(self.params,open(common_path+'/agent_params.p','wb'))
        data_batches = {'l': len(self.memory.data)//50000+1}
        for i in range(0, data_batches['l']):
            if i+1 < data_batches['l']:
                pickle.dump(self.memory.data[50000*i:50000*(i+1)],open(common_path+'/memory'+str(i+1)+'.p','wb'))
            else:
                pickle.dump(self.memory.data[50000*i:-1],open(common_path+'/memory'+str(i+1)+'.p','wb'))
        pickle.dump(data_batches,open(common_path+'/data_batches','wb'))
        # pickle.dump(self.memory,open(common_path+'/memory.p','wb'))
        torch.save(self.critic1.state_dict(), specific_path+'_critic1.pt')
        torch.save(self.critic2.state_dict(), specific_path+'_critic2.pt')
        torch.save(self.baseline.state_dict(), specific_path+'_baseline.pt')
        torch.save(self.baseline_target.state_dict(), specific_path+'_baseline_target.pt')
        torch.save(self.actor.state_dict(), specific_path+'_actor.pt')
        
        if self.hierarchical:
            torch.save(self.concept_model.state_dict(), specific_path+'_concept_model.pt')
            pickle.dump(self.concept_model.prior, open(specific_path+'_prior.pt','wb'))
            pickle.dump(self.transition_model, open(specific_path+'_transition_model.pt','wb'))
            # torch.save(self.concept_critic.state_dict(), specific_path+'_concept_critic.pt')
            torch.save(self.reward_model.state_dict(), specific_path+'_reward_model.pt')
            if self.multitask:
                pickle.dump(self.meta_actor,open(specific_path+'_meta_actor.p','wb'))
                pickle.dump(self.meta_critic,open(specific_path+'_meta_critic.p','wb'))
                # pickle.dump(self.upper_memory,open(common_path+'/upper_memory.p','wb'))
                upper_data_batches = {'l': len(self.upper_memory.data)//15000+1}
                for i in range(0, upper_data_batches['l']):
                    if i+1 < upper_data_batches['l']:
                        pickle.dump(self.upper_memory.data[15000*i:15000*(i+1)],open(common_path+'/upper_memory'+str(i+1)+'.p','wb'))
                    else:
                        pickle.dump(self.upper_memory.data[15000*i:-1],open(common_path+'/upper_memory'+str(i+1)+'.p','wb'))
                pickle.dump(upper_data_batches,open(common_path+'/upper_data_batches','wb'))

                # for i in range(0, self.n_tasks):
                #     pickle.dump(self.upper_memory[i],open(common_path+'/upper_memory_'+str(i)+'.p','wb'))
    
    def load(self, common_path, specific_path, load_memory=True, load_upper_memory=True, transfer=False):
        if load_memory and not transfer: 
            data_batches = pickle.load(open(common_path+'/data_batches','rb'))
            pointer = 0
            for i in range(0, data_batches['l']):
                data = pickle.load(open(common_path+'/memory'+str(i+1)+'.p','rb'))
                self.memory.data += data
                pointer += len(data)
            self.memory.pointer = pointer % self.memory.capacity
            #self.memory = pickle.load(open(common_path+'/memory.p','rb'))
        # self.memory.data = self.memory.data[:self.memory.capacity]
        # self.memory.pointer = 0

        self.actor.load_state_dict(torch.load(specific_path+'_actor.pt'))
        self.actor.eval()

        if not transfer:
            self.critic1.load_state_dict(torch.load(specific_path+'_critic1.pt'))
            self.critic2.load_state_dict(torch.load(specific_path+'_critic2.pt'))
            self.baseline.load_state_dict(torch.load(specific_path+'_baseline.pt'))
            self.baseline_target.load_state_dict(torch.load(specific_path+'_baseline_target.pt'))

            self.critic1.eval()
            self.critic2.eval()
            self.baseline.eval()
            self.baseline_target.eval()
        
        if self.hierarchical:
            self.concept_model.load_state_dict(torch.load(specific_path+'_concept_model.pt'))
            # self.concept_critic.load_state_dict(torch.load(specific_path+'_concept_critic.pt'))

            self.concept_model.eval()
            # self.concept_critic.eval()

            if not transfer:
                self.concept_model.prior = pickle.load(open(specific_path+'_prior.pt','rb'))
                self.transition_model = pickle.load(open(specific_path+'_transition_model.pt','rb'))            
                self.reward_model.load_state_dict(torch.load(specific_path+'_reward_model.pt'))

                if self.multitask:
                    self.meta_actor = pickle.load(open(specific_path+'_meta_actor.p','rb'))
                    self.meta_critic1 = pickle.load(open(specific_path+'_meta_critic.p','rb'))
                    self.meta_critic2 = self.meta_critic1.copy()
                    if load_upper_memory:  
                        # self.upper_memory = pickle.load(open(common_path+'/upper_memory.p','rb'))
                        upper_data_batches = pickle.load(open(common_path+'/upper_data_batches','rb'))
                        upper_pointer = 0
                        for i in range(0, upper_data_batches['l']):
                            upper_data = pickle.load(open(common_path+'/upper_memory'+str(i+1)+'.p','rb'))
                            self.upper_memory.data += upper_data
                            upper_pointer += len(upper_data)
                        self.upper_memory.pointer = upper_pointer % self.upper_memory.capacity
            
                self.reward_model.eval()

#----------------------------------------------
#
#                  System class
#
#----------------------------------------------
class System:
    def __init__(self, params, agent_params={}):
        
        self.params = params
        default_params = {
                            'seed': 1000,
                            'env_names': ['Hopper-v2'],
                            'hierarchical': True,
                            'env_steps': 1, 
                            'grad_steps': 1, 
                            'init_steps': 10000,
                            'beta_coefficient': 1.0,
                            'iss_threshold': 0.96, 
                            'batch_size': 256, 
                            'hard_start': False, 
                            'original_state': True,            
                            'render': True, 
                            'embedded_envs': False,
                            'reset_when_done': True, 
                            'store_video': False, 
                            'basic_epsds': 0,
                            'n_basic_tasks': 1,
                            'max_episode_steps': 1000
                        }

        for key, value in default_params.items():
            if key not in self.params.keys():
                self.params[key] = value

        self.seed = self.params['seed']
        set_seed(self.seed)
        self.env_names = self.params['env_names']
        self.n_tasks = len(self.env_names)        
        self.env_steps = self.params['env_steps']
        self.grad_steps = self.params['grad_steps']
        self.init_steps = self.params['init_steps']
        self.batch_size = self.params['batch_size']
        self.hard_start = self.params['hard_start']
        self.original_state = self.params['original_state']
        self.render = self.params['render']
        self.store_video = self.params['store_video']
        self.reset_when_done = self.params['reset_when_done']
        self.embedded_envs = self.params['embedded_envs']
        self.hierarchical = self.params['hierarchical'] 
        self.basic_epsds = self.params['basic_epsds']
        self.n_basic_tasks = self.params['n_basic_tasks']
        self.beta_coefficient = self.params['beta_coefficient']
        self.iss_threshold = self.params['iss_threshold']
        self._max_episode_steps = self.params['max_episode_steps']

        self.epsd_counter = 0
        self.multitask = self.n_tasks > 0 or self.embedded_envs

        self.set_envs()
        if self.embedded_envs:
            self.task = self.envs[0]._task
        else:
            self.task = 0                     
        
        self.s_dim = self.envs[0].observation_space.shape[0]
        self.a_dim = self.envs[0].action_space.shape[0]        
        if not self.original_state:
            if 'Pendulum-v0' in self.env_names:
                self.s_dim -= 1
            elif 'Ant-v3' in self.env_names:
                self.s_dim = 28        
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = self.s_dim*2 + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 2

        self.min_action = self.envs[0].action_space.low[0]
        self.max_action = self.envs[0].action_space.high[0]

        # if len(intersection(self.env_names, ['Ant-v3','AntCrossMaze-v2', 'AntRandomDirection-v3'])) > 0:
        #     min_c_s *= 2.0
        # elif 'HalfCheetah-v2' in self.env_names:
        #     min_c_s *= 1.5

        self.agent = Agent(self.s_dim, self.a_dim, agent_params) 

    def set_envs(self):
        self.envs = []
        for i in range(0, self.n_tasks):                    
            self.envs.append(gym.make(self.env_names[i]).unwrapped)
            print("Created env "+self.env_names[i])
            self.envs[i].reset()
            self.envs[i].seed(self.seed)        
            self.envs[i]._max_episode_steps = self._max_episode_steps
            self.envs[i].rgb_rendering_tracking = True
    
    @property
    def task_modulo(self):
        modulo = self.n_tasks
        if self.epsd_counter < self.basic_epsds:
            modulo = self.n_basic_tasks
        return modulo

    def reset(self, change_env=False):        
        if self.embedded_envs:
            if self.env_names[self.task] == 'Pendulum-v0' and self.hard_start:
                self.envs[self.task].state = np.array([-np.pi,0.0])
            else:
                self.envs[self.task].reset()
                self.task = self.envs[0]._task
        else:
            if change_env:
                self.task = (self.task+1) % self.task_modulo
            if self.env_names[self.task] == 'Pendulum-v0' and self.hard_start:
                self.envs[self.task].state = np.array([-np.pi,0.0])
            else:
                self.envs[self.task].reset()
            
            # if self.hierarchical:
            self.agent.reset_upper_level()
    
    def get_obs(self):
        if self.original_state:
            state = self.envs[self.task]._get_obs().copy()
        else:
            if self.env_names[self.task] == 'Pendulum-v0':            
                state = self.envs[self.task].state.copy().reshape(-1) 
                state[0] = normalize_angle(state[0])
            elif self.env_names[self.task] == 'Ant-v3':
                state = self.envs[self.task]._get_obs()[:28]
        return state

     
    # def initialization(self):
    def initialization(self, epsd_steps):         
        self.reset()
        self.epsd_counter += 1
        total_r = 0.0
        epsd_step = 0
        for init_step in range(0, self.init_steps):
            epsd_step += 1           
            event = self.interaction_init(epsd_step)
            r = event[self.sa_dim]
            done = event[self.sars_dim]
            total_r += r/self.n_tasks
            if done or (init_step+1) % (epsd_steps) == 0:
                epsd_step = 0
                self.reset(change_env=True)
            if self.render:
                self.envs[self.task].render()                        
        print("Finished initialization, av. reward = %.4f" % (total_r))

    def interaction_init(self, epsd_step):  
        event = np.empty(self.t_dim)
        state = self.get_obs()
        action = self.agent.act(state, self.task, explore=True)[0]
        scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
        reward, done = self.envs[self.task].step(scaled_action)[1:3]  
        done = done and self.reset_when_done
        next_state = self.get_obs()
        
        event[:self.s_dim] = state
        event[self.s_dim:self.sa_dim] = action
        event[self.sa_dim] = reward
        event[self.sa_dim+1:self.sars_dim] = next_state
        event[self.sars_dim] = float(done)
        event[self.sarsd_dim+1] = self.task
        
        if self.multitask:
            event[self.sarsd_dim] = reward # info['reward_goal']  
        else:
            event[self.sarsd_dim] = reward
        
        # if self.hierarchical:
        #   self.agent.update_upper_level(event[self.sarsd_dim], done, self.task, epsd_step>=self.envs[self.task]._max_episode_steps, state, action, action_llhood)    
        
        self.agent.memorize(event)   
        return event

    def interaction(self, learn_upper=True, remember=True, init=False, explore=True, epsd_step=0, learn_lower=True):  
        event = np.empty(self.t_dim)
        state = self.get_obs()

        for env_step in range(0, self.env_steps):
            action, action_llhood = self.agent.act(state, self.task, explore=explore)
            scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
            _, reward, done, info = self.envs[self.task].step(scaled_action)
            # if (epsd_step*self.env_steps+1) >= 1000:
            #     print("Done "+str(done))
            done = done and self.reset_when_done # must be changed if done == True when time == max_time
            next_state = self.get_obs()                            

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.sars_dim] = next_state
            event[self.sars_dim] = float(done)
            event[self.sarsd_dim+1] = self.task
        
            if self.multitask:
                event[self.sarsd_dim] = reward # info['reward_goal']                  
            else:
                event[self.sarsd_dim] = reward    
            
            if remember:
                self.agent.memorize(event)

            if self.hierarchical:
                self.agent.update_upper_level(reward, done, self.task, (epsd_step*self.env_steps+env_step+1)>=self.envs[self.task]._max_episode_steps, 
                    state, action, action_llhood, remember=remember, learn=learn_upper)

            if done:                
                break

            if env_step < self.env_steps-1:
                state = np.copy(next_state)
        
        if learn_lower and not init:
            for _ in range(0, self.grad_steps):
                self.agent.learn_lower()

        return event, done
    
    def train_agent(self, tr_epsds, epsd_steps, initialization=True, eval_epsd_interval=10, eval_epsds=12, iter_=0, save_progress=True, common_path='', 
        rewards=[], goal_rewards=[], metrics=[], learn_lower=True):        
        if self.render:
            self.envs[self.task].render()

        if initialization:
            # self.initialization()
            self.initialization(epsd_steps)

        n_done = 0
        # rewards = []
        # if self.multitask:
        #     goal_rewards = []

        for epsd in range(0, tr_epsds):
            self.epsd_counter += 1
            if epsd == 0:
                self.reset(change_env=False)
            else:
                self.reset(change_env=True)
            
            for epsd_step in range(0, epsd_steps):
                if len(self.agent.memory.data) < self.batch_size:
                    done = self.interaction(learn_upper=False, learn_lower=False, epsd_step=epsd_step)[1]
                else:
                    done = self.interaction(learn_upper=True, learn_lower=learn_lower, epsd_step=epsd_step)[1]

                if self.render:
                    self.envs[self.task].render()

                if done:
                    n_done += 1
                    self.reset(change_env=False)
                
                if n_done >= 5:
                    current_task = self.task
                    self.eval_agent(1, act_randomly=False, iter_=iter_, print_space=False)
                    self.reset()
                    self.task = current_task
                    n_done = 0
            
            if (epsd+1) % eval_epsd_interval == 0:
                if self.hierarchical:
                    if self.multitask:
                        r, gr, _, m = self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[:4]
                        goal_rewards.append(gr)
                        if save_progress:
                            np.savetxt(common_path + '/mean_rewards_goal.txt', np.array(goal_rewards))
                    else:
                        r, _, m = self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[:3]
                    metrics.append(m)
                    if save_progress:
                        np.savetxt(common_path + '/metrics.txt', np.array(metrics))
                else:
                    if self.multitask:
                        r, gr = self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[:2]
                        goal_rewards.append(gr)
                        if save_progress:
                            np.savetxt(common_path + '/mean_rewards_goal.txt', np.array(goal_rewards))
                    else:
                        rewards.append(self.eval_agent(eval_epsds, act_randomly=False, iter_=iter_ + (epsd+1) // eval_epsd_interval)[0])
                rewards.append(r)
                if save_progress:
                    specific_path = common_path + '/' + str(iter_ + (epsd+1) // eval_epsd_interval)
                    self.save(common_path, specific_path)
                    np.savetxt(common_path + '/mean_rewards.txt', np.array(rewards))
              
        if self.multitask:
            return np.array(rewards).reshape(-1), np.array(goal_rewards).reshape(-1)
        else:      
            return np.array(rewards).reshape(-1)      
    

    def eval_agent(self, eval_epsds, act_randomly=False, iter_=0, start_render=False, print_space=True, specific_path='0', max_epsd=-1):   
        if start_render:
            self.envs[self.task].render()
          
                  
        
        events = []
        rewards = []
        epsd_lenghts = []
        min_epsd_reward = 1e6
        max_epsd_reward = -1e6
        
        if self.multitask:
            goal_rewards = [] 
            min_epsd_goal_reward = 1e6
            max_epsd_goal_reward = -1e6
        
        if self.hierarchical:
            uniques = []
            HS = []
            HS_s = []
            ISs = []
            Ha_As = []
            HA_as = []
            HA_s = []
            HA_S = []
            HA = []
            HnS_AT = []
            HnS_SAT = []
            InSS_AT = []
            HR_ST = []
            HR_T = []
            IRS_T = []
            D1 = []
            D2 = []
            D3 = []
            unique_average = 0
            HS_average = 0
            HS_s_average = 0
            ISs_average = 0
            HR_ST_average = 0
            HR_T_average = 0
            IRS_T_average = 0
            InSS_AT_average = 0
            HA_S_average = 0
            HA_average = 0
            Ha_As_average = 0
            HA_as_average = 0
            HA_s_average = 0
            HnS_AT_average = 0
            HnS_SAT_average = 0
            D1_average = 0
            D2_average = 0
            D3_average = 0

        for epsd in range(0, eval_epsds):

            if self.store_video:
                if self.env_names[self.task] == 'Pendulum-v0':
                    video = VideoWriter(specific_path + '_' + str(self.task) + '_' + str(epsd) + '.avi', fourcc, float(FPS), (500, 500))
                else:
                    video = VideoWriter(specific_path + '_' + str(self.task) + '_' + str(epsd) + '.avi', fourcc, float(FPS), (width, height))

            epsd_reward = 0.0
            if self.multitask:
                epsd_goal_reward = 0.0            

            self.reset(change_env=True)
            
            if max_epsd < 0:
                max_epsd_task = self.envs[self.task]._max_episode_steps
            else:
                max_epsd_task = max_epsd

            for eval_step in itertools.count(0):            
                event = self.interaction(learn_upper=False, learn_lower=False, explore=act_randomly, remember=False, epsd_step=eval_step)[0]                

                if self.store_video:
                    if self.env_names[self.task] == 'Pendulum-v0':
                        img = self.envs[self.task].render('rgb_array')
                    else:
                        img = self.envs[self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif self.render:
                    self.envs[self.task].render()   

                r = event[self.sa_dim]
                if self.multitask:
                    R = event[self.sarsd_dim]
                done = event[self.sars_dim]

                epsd_reward += r
                if self.multitask:
                    epsd_goal_reward += R
                events.append(event)

                if done or (eval_step + 1 >= max_epsd_task):
                    epsd_lenghts.append(eval_step + 1)
                    break               

            if self.hierarchical:
                metrics = self.agent.estimate_metrics()
                uniques.append(metrics['n concepts'])
                HS.append(metrics['H(S)'])
                HS_s.append(metrics['H(S|s)'])
                ISs.append(metrics['I(S:s)'])
                Ha_As.append(metrics['H(a|A,s)'])
                HA_as.append(metrics['H(A|a,s)'])
                HA_s.append(metrics['H(A|s)'])
                HA_S.append(metrics['H(A|S)'])
                HA.append(metrics['H(A)'])
                HnS_AT.append(metrics['H(nS|A,T)'])
                HnS_SAT.append(metrics['H(nS|S,A,T)'])
                InSS_AT.append(metrics['I(nS:S|A,T)'])
                HR_ST.append(metrics['H(R|S,T)'])
                HR_T.append(metrics['H(R|T)'])
                IRS_T.append(metrics['I(R:S|T)'])
                D1.append(metrics['D1'])
                D2.append(metrics['D2'])
                D3.append(metrics['D3'])
           
            rewards.append(epsd_reward)
            min_epsd_reward = np.min([epsd_reward, min_epsd_reward])
            max_epsd_reward = np.max([epsd_reward, max_epsd_reward])
            average_reward = np.array(rewards).mean()  

            if self.multitask:
                goal_rewards.append(epsd_goal_reward)            
                min_epsd_goal_reward = np.min([epsd_goal_reward, min_epsd_goal_reward])
                max_epsd_goal_reward = np.max([epsd_goal_reward, max_epsd_goal_reward])           
                average_goal_reward = np.array(goal_rewards).mean()

            if self.hierarchical:
                unique_average += (uniques[-1] - unique_average)/(epsd+1)
                ISs_average += (ISs[-1] - ISs_average)/(epsd+1)
                HR_ST_average += (HR_ST[-1] - HR_ST_average)/(epsd+1)
                HR_T_average += (HR_T[-1] - HR_T_average)/(epsd+1)
                IRS_T_average += (IRS_T[-1] - IRS_T_average)/(epsd+1)
                InSS_AT_average += (InSS_AT[-1] - InSS_AT_average)/(epsd+1)
                HS_average += (HS[-1] - HS_average)/(epsd+1)
                HS_s_average += (HS_s[-1] - HS_s_average)/(epsd+1)
                Ha_As_average += (Ha_As[-1] - Ha_As_average)/(epsd+1)
                HA_as_average += (HA_as[-1] - HA_as_average)/(epsd+1)
                HA_s_average += (HA_s[-1] - HA_s_average)/(epsd+1)
                HA_S_average += (HA_S[-1] - HA_S_average)/(epsd+1)
                HA_average += (HA[-1] - HA_average)/(epsd+1)
                HnS_AT_average += (HnS_AT[-1] - HnS_AT_average)/(epsd+1)
                HnS_SAT_average += (HnS_SAT[-1] - HnS_SAT_average)/(epsd+1)
                D1_average += (D1[-1] - D1_average)/(epsd+1)
                D2_average += (D2[-1] - D2_average)/(epsd+1)
                D3_average += (D3[-1] - D3_average)/(epsd+1)

            if self.hierarchical:
                stdout.write("Iter %i, epsd %i, u: %.1f, I(s:S): %.3f, I(r:S):%.2f, I(nS:S): %.3f, H(a|A,s): %.2f, H(A|a,s): %.2f, H(A|s): %.2f, H(A|S): %.2f, H(A): %.2f, D1: %.2e,  D2: %.1e, D3: %.2e, min r: %i, max r: %i, mean r: %i, ep r: %i\r " %
                    (iter_, (epsd+1), unique_average, ISs_average, IRS_T_average, InSS_AT_average, Ha_As_average, HA_as_average, HA_s_average, HA_S_average, HA_average, D1_average, D2_average, D3_average, min_epsd_reward//1, max_epsd_reward//1, average_reward//1, epsd_reward//1))
                stdout.flush() 
            else:
                stdout.write("Iter %i, epsd %i, min r: %.1f, max r: %.1f, mean r: %.2f, epsd r: %.1f\r " %
                    (iter_, (epsd+1), min_epsd_reward, max_epsd_reward, average_reward, epsd_reward))
                stdout.flush()            

        if print_space:    
            print("")
            # if self.hierarchical:
            #     writer.add_scalar('Eval metrics/H(A|as)', Ha_As_average, iter_)
            #     
            #     writer.add_scalar('Eval metrics/H(S)', HS_average, iter_)
            #     writer.add_scalar('Eval metrics/H(S|s)', HS_s_average, iter_)
            #     writer.add_scalar('Eval metrics/I(S:s)', ISs_average, iter_)
            #     writer.add_scalar('Eval metrics/n_concepts', unique_average, iter_)
            #     
            #     writer.add_scalar('Eval metrics/H(nS|A,T)', HnS_AT_average, iter_)
            #     writer.add_scalar('Eval metrics/H(nS|S,A,T)', HnS_SAT_average, iter_)
            #     writer.add_scalar('Eval metrics/I(nS:S|A,T)', InSS_AT_average, iter_)
            # 
            #     writer.add_scalar('Eval metrics/H(R|S,T)', HR_ST_average, iter_)

        if self.store_video:
            video.release()
        # if self.render:
        #     self.envs[self.task].close()   

        if self.hierarchical:
            metric_vector = np.array([Ha_As_average, HS_average, HS_s_average, ISs_average, unique_average, HnS_AT_average, HnS_SAT_average, InSS_AT_average, HR_ST_average, HR_T_average, IRS_T_average, HA_average, D1_average, D2_average, D3_average, HA_as_average, HA_s_average, HA_S_average]) 
            # sum = self.agent.beta + self.agent.eta + self.agent.nu
            # if ISs_average < self.iss_threshold * np.log(self.agent.n_m_states):                
            #     beta = self.beta_coefficient * self.agent.beta                
            # else:
            #     beta = self.agent.beta / self.beta_coefficient
            # self.agent.eta = self.agent.eta * sum / (sum-self.agent.beta+beta)
            # self.agent.nu = self.agent.nu * sum / (sum-self.agent.beta+beta)
            # self.agent.beta = beta * sum / (sum-self.agent.beta+beta)
            if self.multitask:
                return rewards, goal_rewards, np.array(events), metric_vector, np.array(epsd_lenghts)
            else:
                return rewards, np.array(events), metric_vector, np.array(epsd_lenghts)
        else:
            if self.multitask:
                return rewards, goal_rewards, np.array(events), np.array(epsd_lenghts)
            else:
                return rewards, np.array(events), np.array(epsd_lenghts)
    
    def save(self, common_path, specific_path):
        pickle.dump(self.params,open(common_path+'/params.p','wb'))
        self.agent.save(common_path, specific_path)
    
    def load(self, common_path, specific_path, load_memory=True, load_upper_memory=True, transfer=False):
        self.agent.load(common_path, specific_path, load_memory=load_memory, load_upper_memory=load_upper_memory, transfer=transfer)

    
    # def generate_representative_frames(self, n=10, n_test=0):
    #     if not os.path.isdir('representative_frames/test_'+str(n_test)):
    #         os.mkdir('representative_frames/test_'+str(n_test))
    #     self.envs[self.task].render()
    #     samples = self.agent.concept_model.sample(n).detach().cpu().numpy().transpose(1,0,2).reshape([-1,self.s_dim])
    #     if self.type == 'Hopper-v2':
    #         positions = samples[:,:5]
    #         positions[:,0] = np.clip(samples[:,0], 1.0, np.infty) # height must be above ground level
    #         # positions[:,0] += positions[:,0].min()
    #         velocities = np.clip(samples[:,5:], -10, 10)
    #     for S in range(0, self.agent.n_m_states):
    #         for i in range(0,n):           
    #             qpos = np.concatenate([(np.random.rand(1)-0.5)*40,positions[n*S+i,:]]) # pos[0] is a random x position
    #             qpos = np.concatenate([np.zeros(1),positions[n*S+i,:]]) # pos[0] is a random x position                
    #             qvel = velocities[n*S+i,:]
    #             p = self.envs[self.task].init_qpos
    #             v = self.envs[self.task].init_qvel
    #             self.envs[self.task].set_state(qpos, qvel)
    #             rgb_array = self.envs[self.task].render('rgb_array',1024,768)
    #             img = Image.fromarray(rgb_array, 'RGB')
    #             img.save('representative_frames/test_'+str(n_test)+'/concept_'+str(S)+'_'+str(i)+'.png')
    #     self.envs[self.task].close()
