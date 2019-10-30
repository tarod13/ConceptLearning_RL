import gym
import torch
import numpy as np
from scipy.special import gamma as f_gamma

import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.nn.utils import clip_grad_norm_

from nets import (Memory, v_valueNet, q_valueNet, policyNet, conditionalEncoder, mixtureConceptModel, encoderConceptModel, 
                    rNet, RNet, v_parallel_valueNet, q_parallel_valueNet, nextSNet, conceptLossNet)
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
                            'alpha_upper_level': 10.0,
                            'upper_level_period': 3, 
                            'concept_model_type': 'encoder',
                            'policy_batch_size': 256,
                            'concept_batch_size': 256,  
                            'soft_lr': 5e-3, 
                            'memory_capacity': 500000, 
                            'gamma': 0.99, 
                            'alpha': 1.0, 
                            'beta': 1.0, 
                            'eta': 1.0, 
                            'mu': 1.0, 
                            'nu': 1.0,
                            'zeta': 1.0,
                            'xi': 10.0, 
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
                            'tau_upper_level': 1.0,
                            'coefficient_upper_annealing': 0.999993,
                            'min_alpha_upper_level': 5e-2
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
        self.tau_upper_level = self.params['tau_upper_level']

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

        # Net parameters
        self.policy_batch_size = self.params['policy_batch_size']
        self.concept_batch_size = self.params['concept_batch_size']
        self.soft_lr = self.params['soft_lr']
        self.clip_value = self.params['clip_value']
        
        # Metric weights
        self.alpha = self.params['alpha']
        self.beta = self.params['beta']        
        self.eta = self.params['eta']
        self.nu = self.params['nu']
        self.mu = self.params['mu']
        self.zeta = self.params['zeta']        
        
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
            self.concept_critic = conceptLossNet(s_dim, self.n_m_states).to(device)
            self.evolution_model = nextSNet(s_dim, self.n_m_states, self.n_m_actions, n_tasks=self.n_tasks).to(device)
            self.reward_model = rNet(s_dim, self.n_m_actions, n_tasks=self.n_tasks, lr=self.params['r_lr']).to(device)
            # self.reward_model = RNet(n_m_states, n_tasks=n_tasks, lr=r_lr).to(device)
            self.meta_critic = np.ones([self.n_tasks, self.n_m_states, self.n_m_actions])/self.alpha
            if self.multitask:
                self.meta_actor = np.ones([self.n_tasks, self.n_m_states, self.n_m_actions])/self.n_m_actions
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

    def meta_learning(self, event):
        S = int(event[0])
        A = int(event[1])
        R = event[2]
        nS = int(event[3])
        d = event[5]
        T = int(event[6])
        # P_S = event[7:7+self.n_m_states]
        # P_nS = event[7+self.n_m_states:]
        
        # # Update value estimation
        # Q_nS = self.meta_critic[T, nS, :].copy()
        # Pi_nS = self.meta_actor[T, nS, :].copy()
        # V_nS = (Pi_nS * (Q_nS - self.alpha_upper_level * np.log(Pi_nS+1e-6))).sum()
        # self.meta_critic[T, S, A] = R + self.gamma*(1-d)*V_nS
        # # Q_T = self.meta_critic[T, :, :].copy()
        # # Pi_T = self.meta_actor[T, :, :].copy()
        # # V_nS = ((Pi_T * (Q_T - self.alpha_upper_level * np.log(Pi_T+1e-20))).sum(1) * P_nS).sum()
        # # self.meta_critic[T, :, A] += P_S.copy()*(R + self.gamma*(1-d)*V_nS - self.meta_critic[T, :, A])
        
        # # Update policy
        # Q_S = self.meta_critic[T, S, :].copy()
        # weighted_Q_S = self.tau_upper_level * Q_S / self.alpha_upper_level
        # weighted_Q_S -= weighted_Q_S.max()
        # expQ_S = np.exp(weighted_Q_S)
        # Z_S = expQ_S.sum()
        # assert Z_S >= 1.0, 'distribution calculated incorrectly'
        # Pi_S = expQ_S/Z_S
        # self.meta_actor[T, S, :] = Pi_S.copy() / Pi_S.sum()

        # Update value estimation
        Q_nS = self.meta_critic[T, nS, :]
        Pi_nS = self.meta_actor[T, nS, :]
        V_nS = (Pi_nS * (Q_nS - self.alpha_upper_level * np.log(Pi_nS+1e-20))).sum()
        self.meta_critic[T, S, A] = R + self.gamma*(1-d)*V_nS
        
        # Update policy
        Q = self.meta_critic.copy()
        Q = Q - Q.max(2, keepdims=True)
        expQ = np.exp(Q/self.alpha_upper_level)
        Z = expQ.sum(2, keepdims=True)
        self.meta_actor = expQ/Z

        # Update prior
        self.concept_model.update_prior(S, T)
        self.update_transition_model(S, A, nS, T)
    
    def update_transition_model(self, S, A, nS, T):
        self.transition_model[T,S,A,nS] += 1.0
        self.transition_model[T,S,A,:] *= self.prior_n/(self.prior_n+1)

    #     # Update prior
    #     self.concept_model.update_prior(S, T, P_S, self.model_update_method)
    #     self.update_transition_model(S, A, nS, T, P_nS)
    
    # def update_transition_model(self, S, A, nS, T, P_nS):
    #     if self.model_update_method == 'discrete':
    #         self.transition_model[T,S,A,nS] += 1.0
    #     else:
    #         self.transition_model[T,S,A,:] += torch.FloatTensor(P_nS).to(device)
    #     self.transition_model[T,S,A,:] *= self.prior_n/(self.prior_n+1)

    def high_level_decision(self, task, S, explore):
        if (self.bottom_level_epsds % self.upper_level_period) == 0:
            policy = self.meta_actor[task, S, :].reshape(-1)
            if explore:
                A = np.random.multinomial(1, policy/policy.sum()).argmax()
            else:
                A = policy.argmax()
            self.m_action = A
        else:
            A = self.m_action
        return A

    def concept_inference(self, s, explore=False):
        if (self.bottom_level_epsds % self.upper_level_period) == 0:
            # S, PS = self.concept_model.sample_m_state(torch.from_numpy(s).float().to(device), explore=explore)
            S = self.concept_model.sample_m_state(torch.from_numpy(s).float().to(device), explore=explore)
            self.m_state = S
            # self.posterior = PS.detach().cpu().numpy()
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
    
    def update_upper_level(self, r, done, task, completed_episode, state, action, action_llhoods):
        self.cumulative_reward += r
        self.action_llhoods += action_llhoods.copy()
        self.past_states.append(state.copy())
        self.past_actions.append(action.copy())
        self.time_flow(done, task, completed_episode, r, state, action, action_llhoods)

    def time_flow(self, done, task, completed_episode, r, state, action, action_llhoods): 
        to_learn_complete = self.stored and (self.bottom_level_epsds % self.upper_level_period) == 0       
        if to_learn_complete:
            if self.hierarchical:
                # learning_event = np.empty(7+2*self.n_m_states)
                learning_event = np.empty(7)
                learning_event[0] = self.past_m_state
                learning_event[1] = self.past_m_action
                learning_event[2] = self.cumulative_reward - r
                learning_event[3] = self.m_state
                learning_event[4] = self.m_action
                learning_event[5] = float(done)
                learning_event[6] = task
                # learning_event[7:7+self.n_m_states] = self.past_posterior.copy()
                # learning_event[7+self.n_m_states:] = self.posterior.copy()
                self.meta_learning(learning_event.copy())

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
            if self.upper_level_annealing:
                self.alpha_upper_level = np.max([self.coefficient_upper_annealing * self.alpha_upper_level, self.min_alpha_upper_level])

        if not self.stored and (self.bottom_level_epsds % self.upper_level_period) == 0:
            self.past_m_state = self.m_state
            self.past_m_action = self.m_action
            # self.past_posterior = self.posterior.copy()
            # self.past_states = state.copy()
            self.stored = True
        
        if done and not to_learn_complete and self.hierarchical:
            # event = np.empty(7+2*self.n_m_states)
            event = np.empty(7)
            event[0] = self.m_state
            event[1] = self.m_action
            event[2] = self.cumulative_reward
            event[3] = self.m_state
            event[4] = self.m_action
            event[5] = float(done)
            event[6] = task
            # event[7:7+self.n_m_states] = self.past_posterior.copy()
            # PS = self.concept_model.sample_m_state(torch.from_numpy(state).float().to(device), explore=True)[1]
            # event[7+self.n_m_states:] = PS.detach().cpu().numpy().copy()
            self.meta_learning(event.copy())
            # self.train_reward_model(task, self.past_states, self.cumulative_reward)
            # self.train_evolution_model(task, self.past_states, state, self.m_action)     

            # storing_event = np.empty(2*self.s_dim+3)
            # storing_event[:self.s_dim] = self.past_states.copy()
            # storing_event[self.s_dim:2*self.s_dim] = state.copy()
            # storing_event[-3] = self.m_action
            # storing_event[-2] = task
            # storing_event[-1] = self.cumulative_reward
            # self.memorize_in_upper_level(storing_event)               

        self.bottom_level_epsds += 1

        if done or completed_episode:
            self.reset_upper_level()       

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

            # Optimize v network
            if self.hierarchical:
                S_batch_off, posterior_off = self.concept_model.sample_m_state_and_posterior(s_batch)[0:2]                
                if self.multitask:  
                    policy = torch.from_numpy(self.meta_actor)[T_batch,:,:].float().to(device)
                    policy_off = self.meta_actor[T_batch.astype(int),S_batch_off.numpy(),:]
                    assert len(policy_off.shape) == 2, 'Wrong size'
                    assert policy_off.shape[0] == S_batch_off.size(0), 'Wrong size'
                    assert policy_off.shape[1] == self.n_m_actions, 'Wrong size'
                    A_batch_off = torch.from_numpy(vectorized_multinomial(policy_off)).long().to(device)                           
                    _, a_batch_off, a_conditional_llhood_off = self.actor.sample_action_and_llhood_pairs(s_batch[:,:self.s_dim], A_batch_off)
                    A_probability_off =  torch.einsum('ijk,ij->ik', policy, posterior_off)                                    
                else:
                    A_batch_off = S_batch_off.clone()
                    _, a_batch_off, a_conditional_llhood_off = self.actor.sample_action_and_llhood_pairs(s_batch[:,:self.s_dim], A_batch_off)
                    A_probability_off = posterior_off.detach() 
                a_log_posterior_unnormalized = a_conditional_llhood_off + torch.log(A_probability_off + 1e-12)
                a_llhood_off = torch.logsumexp(a_log_posterior_unnormalized+1e-12, dim=1, keepdim=True)
            else:
                A_batch_off = torch.zeros(s_batch.size(0),).long().to(device)
                A_probability_off = torch.ones(s_batch.size(0),).long().to(device)
                _, a_batch_off, a_conditional_llhood_off = self.actor.sample_action_and_llhood_pairs(s_batch[:,:self.s_dim], A_batch_off)
                a_llhood_off =  a_conditional_llhood_off

            if not only_metrics:
                q1_off = self.critic1(s_batch.detach(), a_batch_off)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                q2_off = self.critic2(s_batch.detach(), a_batch_off)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                q_off = torch.min(q1_off, q2_off)
                assert q_off.size(1) == 1, 'Wrong size'
            
                v = self.baseline(s_batch)[np.arange(s_batch.size(0)), T_batch].view(-1,1)
                assert v.size(1) == 1, 'Wrong size'
                v_approx = q_off - self.alpha*a_llhood_off
                assert a_llhood_off.size(1) == 1, 'Wrong size'
                assert v_approx.size(1) == 1, 'Wrong size'       

            if self.hierarchical:
                a_conditional_A_entropy = -a_conditional_llhood_off[np.arange(s_batch.size(0)), A_batch_off].view(-1,1)
                if not only_metrics:        
                    v_approx = v_approx - self.mu*a_conditional_A_entropy
                
            if not only_metrics:
                v_loss = self.baseline.loss_func(v, v_approx.detach())
                self.baseline.optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.baseline.parameters(), self.clip_value)
                self.baseline.optimizer.step()
                updateNet(self.baseline_target, self.baseline, self.soft_lr)

                # Optimize skill network
                pi_loss = (-v_approx).mean()
                self.actor.optimizer.zero_grad()
                pi_loss.backward()
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()

        else:
            a_llhood_off = torch.zeros(1).to(device)  
            a_conditional_A_entropy = torch.zeros(1).to(device)
            A_probability_off = torch.ones(1,1).to(device)/self.n_m_actions

        if only_metrics:
            marginal_A = A_probability_off.mean(0, keepdim=True) 
            metrics = {
                'H(a|s)': a_llhood_off.mean().detach().cpu().numpy(),
                'H(a|A,s)': a_conditional_A_entropy.mean().detach().cpu().numpy(),
                'H(A|s)': -(A_probability_off * torch.log(A_probability_off + 1e-12)).sum(1, keepdim=True).mean().detach().cpu().numpy(),
                'H(A)': -(marginal_A * torch.log(marginal_A + 1e-12)).sum().detach().cpu().numpy()
            }
            
            return metrics
    
    def learn_reward(self):
        if self.hierarchical and not self.upper_memory.empty():
            upper_batch = self.upper_memory.sample(self.concept_batch_size)
            upper_batch = np.array(upper_batch)
            
            a_llhoods_upper_batch = torch.FloatTensor(upper_batch[:,self.N_s_dim + self.upper_level_period*self.a_dim:-3]).to(device)
            T_upper_batch = upper_batch[:,-2].astype('int')
            R_upper_batch = torch.FloatTensor(upper_batch[:,-1]).unsqueeze(1).to(device)
            s_upper_batch = torch.FloatTensor(upper_batch[:, 0: self.s_dim]).to(device)

            a_llhoods_upper_batch_off = torch.zeros_like(a_llhoods_upper_batch)
            for i in range(0, self.upper_level_period+1):
                s_upper_batch_ = torch.FloatTensor(upper_batch[:, i*self.s_dim:(i+1)*self.s_dim]).to(device)
                if i < self.upper_level_period:
                    a_upper_batch = torch.FloatTensor(upper_batch[:, self.N_s_dim + i*self.a_dim : self.N_s_dim + (i+1)*self.a_dim]).to(device)
                    a_llhoods_upper_batch_off += self.actor.llhoods(s_upper_batch_, a_upper_batch).detach()

            # Optimize reward network
            a_importance_ratios = torch.exp((a_llhoods_upper_batch_off - a_llhoods_upper_batch).clamp(-1e6,10))+1e-10
            R_llhood = self.reward_model.llhood(R_upper_batch, s_upper_batch, T_upper_batch)
            R_model_loss = -(R_llhood * a_importance_ratios.detach()).mean()
            self.reward_model.optimizer.zero_grad()
            R_model_loss.backward()
            clip_grad_norm_(self.reward_model.parameters(), self.clip_value)
            self.reward_model.optimizer.step()


    def learn_concepts(self, only_metrics=False):                                       
        if self.hierarchical and not self.upper_memory.empty():
            upper_batch = self.upper_memory.sample(self.concept_batch_size)
            upper_batch = np.array(upper_batch)
            batch_size = upper_batch.shape[0]

            a_llhoods_upper_batch = torch.FloatTensor(upper_batch[:,self.N_s_dim + self.upper_level_period*self.a_dim:-3]).to(device)
            T_upper_batch = upper_batch[:,-2].astype('int')
            R_upper_batch = torch.FloatTensor(upper_batch[:,-1]).unsqueeze(1).to(device)

            s_upper_batches = []
            a_upper_batches = []
            a_llhoods_upper_batch_off = torch.zeros_like(a_llhoods_upper_batch)
            for i in range(0, self.upper_level_period+1):
                s_upper_batches.append(torch.FloatTensor(upper_batch[:, i*self.s_dim:(i+1)*self.s_dim]).to(device))
                if i < self.upper_level_period:
                    a_upper_batches.append(torch.FloatTensor(upper_batch[:, self.N_s_dim + i*self.a_dim : self.N_s_dim + (i+1)*self.a_dim]).to(device))
                    a_llhoods_upper_batch_off += self.actor.llhoods(s_upper_batches[-1], a_upper_batches[-1])

            baseline = np.log(batch_size)
            
            # I(s:S)
            _, S_upper_posterior, S_upper_log_posterior = self.concept_model.sample_m_state_and_posterior(s_upper_batches[0])
            S_upper_log_marginal = -baseline + torch.logsumexp(S_upper_log_posterior, dim=0).view(-1,1)
            S_entropy = -(torch.exp(S_upper_log_marginal)*S_upper_log_marginal).sum()
            S_conditional_entropy_on_s = -(S_upper_posterior * S_upper_log_posterior).sum(1).mean()
            s_S_mutual_information = S_entropy - S_conditional_entropy_on_s

            a_importance_ratios = torch.exp((a_llhoods_upper_batch_off - a_llhoods_upper_batch).clamp(-1e6,10))+1e-10
            if not only_metrics:
                # Optimize reward network
                R_llhood = self.reward_model.llhood(R_upper_batch, s_upper_batches[0], T_upper_batch)
                R_model_loss = -(R_llhood * a_importance_ratios.detach()).mean()
                self.reward_model.optimizer.zero_grad()
                R_model_loss.backward()
                clip_grad_norm_(self.reward_model.parameters(), self.clip_value)
                self.reward_model.optimizer.step()

            R_lklhood_given_sApT_A = torch.exp(self.reward_model.sample_and_cross_llhood(s_upper_batches[0], T_upper_batch))
            assert len(R_lklhood_given_sApT_A.shape) == 4, 'P(R|s,A,T) is calculated incorrectly. Wrong size.'
            assert R_lklhood_given_sApT_A.size(0) == batch_size, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 0.'
            assert R_lklhood_given_sApT_A.size(1) == batch_size, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 1.'
            assert R_lklhood_given_sApT_A.size(2) == self.n_m_actions, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 2.'
            assert R_lklhood_given_sApT_A.size(3) == self.n_m_actions, 'P(R|s,A,T) is calculated incorrectly. Wrong dim 2.'
            
            policy_upper = self.meta_actor[T_upper_batch, :, :]
            policy_upper = torch.from_numpy(policy_upper).float().to(device)
            R_lklhood_given_sST_A = torch.einsum('ikp,ijtp->ijkt', policy_upper, R_lklhood_given_sApT_A)
            task_mask = (T_upper_batch.reshape(-1,1) == T_upper_batch.reshape(1,-1))*1.0
            task_mask /= task_mask.sum(1, keepdims=True)
            task_mask = torch.from_numpy(task_mask).float().to(device)
            R_lklhood_given_sST_A_proper_T = R_lklhood_given_sST_A * task_mask.unsqueeze(2).unsqueeze(3)            
            R_lklhood_given_ST_A_unnormalized = torch.einsum('ijkt,jk->ikt', R_lklhood_given_sST_A_proper_T, S_upper_posterior)
            S_probability_given_T = self.concept_model.prior[T_upper_batch, :]/self.concept_model.prior_n            
            R_lklhood_given_ST_A = R_lklhood_given_ST_A_unnormalized / (S_probability_given_T.unsqueeze(2) + 1e-10)
            R_lklhood_given_T_A = R_lklhood_given_ST_A_unnormalized.sum(1, keepdim=True)
            
            R_conditional_entropy_on_ST = -(policy_upper * S_upper_posterior.unsqueeze(2) * torch.log(R_lklhood_given_ST_A + 1e-10)).sum((1,2)).view(-1,1)
            R_conditional_entropy_on_T = -(policy_upper * S_upper_posterior.unsqueeze(2) * torch.log(R_lklhood_given_T_A + 1e-10)).sum((1,2)).view(-1,1)  
            R_S_mutual_information = R_conditional_entropy_on_T - R_conditional_entropy_on_ST          
            
            transition_task_mask = torch.zeros(batch_size, self.n_tasks).float().to(device)
            transition_task_mask[np.arange(batch_size), T_upper_batch] = torch.ones(batch_size).float().to(device)
            task_counts = transition_task_mask.sum(0).view(-1,1,1,1).clamp(1,batch_size)
            transition_task_mask = transition_task_mask.view(batch_size,-1,1,1,1)/ task_counts            
            assert torch.all(a_importance_ratios != float('inf')), 'Infinite lklhoods'
            nS_upper_posterior = self.concept_model.sample_m_state_and_posterior(s_upper_batches[-1])[1]
            nS_lklhood_given_SAT_unnormalized = (a_importance_ratios.view(batch_size,1,1,-1,1) * S_upper_posterior.view(batch_size,1,-1,1,1) * 
                                                nS_upper_posterior.view(batch_size,1,1,1,-1) * transition_task_mask).sum(0) 
            nS_lklhood_given_AT = nS_lklhood_given_SAT_unnormalized.sum(1, keepdim=True)
            S_probability_given_T_matrix = self.concept_model.prior.view(self.n_tasks,-1,1,1)/self.concept_model.prior_n
            nSSAT_lklhood = self.transition_model/self.prior_n * torch.from_numpy(self.meta_actor).float().to(device).unsqueeze(3) * S_probability_given_T_matrix/self.n_tasks
            nS_conditional_entropy_on_SAT = -(nSSAT_lklhood * (torch.log(nS_lklhood_given_SAT_unnormalized+1e-10) - torch.log(S_probability_given_T_matrix+1e-10))).sum()
            nS_conditional_entropy_on_AT = -(nSSAT_lklhood * torch.log(nS_lklhood_given_AT+1e-10)).sum()
            nS_S_mutual_information = nS_conditional_entropy_on_AT - nS_conditional_entropy_on_SAT

            S_upper_posteriors = [(S_upper_posterior, S_upper_log_posterior)]
            posterior_inconsistency = torch.zeros_like(S_upper_posterior)
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

            if not only_metrics:
                disentanglement_metric = self.beta*s_S_mutual_information + self.eta*R_S_mutual_information + self.nu*nS_S_mutual_information
            
                # Optimize concept networks              
                concept_model_loss = (inconsistency_metric-disentanglement_metric).mean()
                self.concept_model.optimizer.zero_grad()
                concept_model_loss.backward()
                clip_grad_norm_(self.concept_model.parameters(), self.clip_value)
                self.concept_model.optimizer.step()
                assert torch.all(self.concept_model.l1.weight==self.concept_model.l1.weight), 'Invalid concept model parameters'

        else:
            S_upper_posterior = torch.zeros(1,1).to(device)
            S_entropy = torch.zeros(1).to(device)
            S_conditional_entropy_on_s = torch.zeros(1).to(device)
            s_S_mutual_information = torch.zeros(1).to(device)
            nS_conditional_entropy_on_AT = torch.zeros(1).to(device)
            nS_conditional_entropy_on_SAT = torch.zeros(1).to(device)                    
            nS_S_mutual_information = torch.zeros(1).to(device)
            R_conditional_entropy_on_ST = torch.zeros(1).to(device)
            R_conditional_entropy_on_T = torch.zeros(1).to(device)
            R_S_mutual_information = torch.zeros(1).to(device)                

        if only_metrics:
            metrics = {
                'n concepts': len(np.unique(S_upper_posterior.argmax(1).detach().cpu().numpy())),
                'H(S)': S_entropy.mean().detach().cpu().numpy(),
                'H(S|s)': S_conditional_entropy_on_s.mean().detach().cpu().numpy(),
                'I(S:s)': s_S_mutual_information.mean().detach().cpu().numpy(), 
                'H(R|S,T)': R_conditional_entropy_on_ST.mean().detach().cpu().numpy(),
                'H(R|T)': R_conditional_entropy_on_T.mean().detach().cpu().numpy(),
                'I(R:S|T)': R_S_mutual_information.mean().detach().cpu().numpy(),
                'H(nS|A,T)': nS_conditional_entropy_on_AT.mean().detach().cpu().numpy(),
                'H(nS|S,A,T)': nS_conditional_entropy_on_SAT.mean().detach().cpu().numpy(),
                'I(nS:S|A,T)': nS_S_mutual_information.mean().detach().cpu().numpy()
            }
            
            return metrics

    def estimate_metrics(self):
        with torch.no_grad():
            skill_metrics = self.learn_skills(only_metrics=True)
            concept_metrics = self.learn_concepts(only_metrics=True)
        metrics = {**skill_metrics, **concept_metrics}
        return metrics

    def learn(self):
        if self.parallel_learning or not self.hierarchical:
            self.learn_skills()
            if self.hierarchical:
                self.learn_concepts()            
        else:
            if self.bottom_level_epsds % (self.skill_steps + self.concept_steps) < self.skill_steps:
                self.learn_skills()
                if self.reward_learning == 'always':
                    self.learn_reward()
            else:
                self.learn_concepts()
    
    def save(self, common_path, specific_path):
        self.params['alpha_upper_level'] = self.alpha_upper_level
        pickle.dump(self.params,open(common_path+'/agent_params.p','wb'))
        pickle.dump(self.memory,open(common_path+'/memory.p','wb'))
        torch.save(self.critic1.state_dict(), specific_path+'_critic1.pt')
        torch.save(self.critic2.state_dict(), specific_path+'_critic2.pt')
        torch.save(self.baseline.state_dict(), specific_path+'_baseline.pt')
        torch.save(self.baseline_target.state_dict(), specific_path+'_baseline_target.pt')
        torch.save(self.actor.state_dict(), specific_path+'_actor.pt')
        
        if self.hierarchical:
            torch.save(self.concept_model.state_dict(), specific_path+'_concept_model.pt')
            pickle.dump(self.concept_model.prior, open(specific_path+'_prior.pt','wb'))
            pickle.dump(self.transition_model, open(specific_path+'_transition_model.pt','wb'))
            torch.save(self.concept_critic.state_dict(), specific_path+'_concept_critic.pt')
            torch.save(self.reward_model.state_dict(), specific_path+'_reward_model.pt')
            if self.multitask:
                pickle.dump(self.meta_actor,open(specific_path+'_meta_actor.p','wb'))
                pickle.dump(self.meta_critic,open(specific_path+'_meta_critic.p','wb'))
                pickle.dump(self.upper_memory,open(common_path+'/upper_memory.p','wb'))

                # for i in range(0, self.n_tasks):
                #     pickle.dump(self.upper_memory[i],open(common_path+'/upper_memory_'+str(i)+'.p','wb'))
    
    def load(self, common_path, specific_path, load_memory=True):
        if load_memory:        
            self.memory = pickle.load(open(common_path+'/memory.p','rb'))
        # self.memory.data = self.memory.data[:self.memory.capacity]
        # self.memory.pointer = 0
        self.critic1.load_state_dict(torch.load(specific_path+'_critic1.pt'))
        self.critic2.load_state_dict(torch.load(specific_path+'_critic2.pt'))
        self.baseline.load_state_dict(torch.load(specific_path+'_baseline.pt'))
        self.baseline_target.load_state_dict(torch.load(specific_path+'_baseline_target.pt'))
        self.actor.load_state_dict(torch.load(specific_path+'_actor.pt'))

        self.critic1.eval()
        self.critic2.eval()
        self.baseline.eval()
        self.baseline_target.eval()
        self.actor.eval()
        
        if self.hierarchical:
            self.concept_model.load_state_dict(torch.load(specific_path+'_concept_model.pt'))
            self.concept_model.prior = pickle.load(open(specific_path+'_prior.pt','rb'))
            self.transition_model = pickle.load(open(specific_path+'_transition_model.pt','rb'))
            self.concept_critic.load_state_dict(torch.load(specific_path+'_concept_critic.pt'))
            self.reward_model.load_state_dict(torch.load(specific_path+'_reward_model.pt'))
            if self.multitask:
                self.meta_actor = pickle.load(open(specific_path+'_meta_actor.p','rb'))
                self.meta_critic = pickle.load(open(specific_path+'_meta_critic.p','rb'))
                if load_memory:  
                    self.upper_memory = pickle.load(open(common_path+'/upper_memory.p','rb'))
            
            self.concept_model.eval()
            self.concept_critic.eval()
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
                            'n_basic_tasks': 1
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
            self.envs[i]._max_episode_steps = 1000
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
            
            if self.hierarchical:
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

    # def initialization(self, epsd_steps): 
    def initialization(self):         
        self.reset()
        self.epsd_counter += 1
        average_r = 0.0
        epsd_step = 0
        for init_step in range(0, self.init_steps):
            epsd_step += 1           
            event = self.interaction_init(epsd_step)
            r = event[self.sa_dim]
            done = event[self.sars_dim]
            average_r += (r-average_r)/(init_step+1)
            if done:
                epsd_step = 0
            # elif (init_step+1) % (10*epsd_steps) == 0:
            #     self.reset(change_env=True)
            if self.render:
                self.envs[self.task].render()                        
        print("Finished initialization, av. reward = %.4f" % (average_r))

    def interaction_init(self, epsd_step):  
        event = np.empty(self.t_dim)
        state = self.get_obs()
        action, action_llhood = self.agent.act(state, self.task, explore=True)
        scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
        _, reward, done, info = self.envs[self.task].step(scaled_action)  
        done = done and self.reset_when_done
        next_state = self.get_obs()
        if done:
            self.reset(change_env=True)                   
        
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
        
        if self.hierarchical:
            self.agent.update_upper_level(event[self.sarsd_dim], done, self.task, epsd_step>=self.envs[self.task]._max_episode_steps, state, action, action_llhood)    
        
        self.agent.memorize(event)   
        return event

    def interaction(self, learn=True, remember=True, init=False, explore=True, epsd_step=0):  
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

            if remember and self.hierarchical:
                self.agent.update_upper_level(event[self.sarsd_dim], done, self.task, (epsd_step*self.env_steps+env_step+1)>=self.envs[self.task]._max_episode_steps, state, action, action_llhood)

            if done:                
                break

            if env_step < self.env_steps-1:
                state = np.copy(next_state)
        
        if learn and not init:
            for _ in range(0, self.grad_steps):
                self.agent.learn()

        return event, done
    
    def train_agent(self, tr_epsds, epsd_steps, initialization=True, eval_epsd_interval=10, eval_epsds=12, iter_=0, save_progress=True, common_path='', rewards=[], goal_rewards=[], metrics=[]):        
        if self.render:
            self.envs[self.task].render()

        if initialization:
            self.initialization()
            # self.initialization(epsd_steps)

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
                    done = self.interaction(learn=False, epsd_step=epsd_step)[1]
                else:
                    done = self.interaction(learn=True, epsd_step=epsd_step)[1]

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
    

    def eval_agent(self, eval_epsds, act_randomly=False, iter_=0, start_render=False, print_space=True, specific_path='0'):   
        if start_render:
            self.envs[self.task].render()
          
        if self.store_video:
            if self.env_names[self.task] == 'Pendulum-v0':
                video = VideoWriter(specific_path + '.avi', fourcc, float(FPS), (500, 500))
            else:
                video = VideoWriter(specific_path + '.avi', fourcc, float(FPS), (width, height))          
        
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
            Ha_s = []
            Ha_As = []
            HA_s = []
            HA = []
            HnS_AT = []
            HnS_SAT = []
            InSS_AT = []
            HR_ST = []
            HR_T = []
            IRS_T = []
            unique_average = 0
            HS_average = 0
            HS_s_average = 0
            ISs_average = 0
            HR_ST_average = 0
            HR_T_average = 0
            IRS_T_average = 0
            InSS_AT_average = 0
            Ha_s_average = 0
            HA_s_average = 0
            HA_average = 0
            Ha_As_average = 0
            HnS_AT_average = 0
            HnS_SAT_average = 0

        for epsd in range(0, eval_epsds):
            epsd_reward = 0.0
            if self.multitask:
                epsd_goal_reward = 0.0            

            self.reset(change_env=True)
            
            for eval_step in itertools.count(0):            
                event = self.interaction(learn=False, explore=act_randomly, remember=False, epsd_step=eval_step)[0]                

                if self.store_video:
                    if self.env_names[self.task] == 'Pendulum-v0':
                        img = self.envs[self.task].render('rgb_array')
                    else:
                        img = self.envs[self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if self.render:
                    self.envs[self.task].render()   

                r = event[self.sa_dim]
                if self.multitask:
                    R = event[self.sarsd_dim]
                done = event[self.sars_dim]

                epsd_reward += r
                if self.multitask:
                    epsd_goal_reward += R
                events.append(event)

                if done or (eval_step + 1 >= self.envs[self.task]._max_episode_steps):
                    epsd_lenghts.append(eval_step + 1)
                    break               

            if self.hierarchical:
                metrics = self.agent.estimate_metrics()
                uniques.append(metrics['n concepts'])
                HS.append(metrics['H(S)'])
                HS_s.append(metrics['H(S|s)'])
                ISs.append(metrics['I(S:s)'])
                Ha_s.append(metrics['H(a|s)'])
                Ha_As.append(metrics['H(a|A,s)'])
                HA_s.append(metrics['H(A|s)'])
                HA.append(metrics['H(A)'])
                HnS_AT.append(metrics['H(nS|A,T)'])
                HnS_SAT.append(metrics['H(nS|S,A,T)'])
                InSS_AT.append(metrics['I(nS:S|A,T)'])
                HR_ST.append(metrics['H(R|S,T)'])
                HR_T.append(metrics['H(R|T)'])
                IRS_T.append(metrics['I(R:S|T)'])
           
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
                Ha_s_average += (Ha_s[-1] - Ha_s_average)/(epsd+1)
                Ha_As_average += (Ha_As[-1] - Ha_As_average)/(epsd+1)
                HA_s_average += (HA_s[-1] - HA_s_average)/(epsd+1)
                HA_average += (HA[-1] - HA_average)/(epsd+1)
                HnS_AT_average += (HnS_AT[-1] - HnS_AT_average)/(epsd+1)
                HnS_SAT_average += (HnS_SAT[-1] - HnS_SAT_average)/(epsd+1)

            if self.hierarchical:
                stdout.write("Iter %i, epsd %i, u: %.2f, I(s:S): %.3f, I(r:S):%.3f, I(nS:S): %.3f, H(a:s): %.3f, H(A:s): %.3f, H(A): %.3f, min r: %.1f, max r: %.1f, mean r: %.2f, epsd r: %.1f\r " %
                    (iter_, (epsd+1), unique_average, ISs_average, IRS_T_average, InSS_AT_average, Ha_s_average, HA_s_average, HA_average, min_epsd_reward, max_epsd_reward, average_reward, epsd_reward))
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
            metric_vector = np.array([Ha_As_average, HS_average, HS_s_average, ISs_average, unique_average, HnS_AT_average, HnS_SAT_average, InSS_AT_average, HR_ST_average, HR_T_average, IRS_T_average, Ha_s_average, Ha_s_average-Ha_As_average, HA_s_average, Ha_s_average-HA_s_average, HA_average]) 
            sum = self.agent.beta + self.agent.eta + self.agent.nu
            if ISs_average < self.iss_threshold * np.log(self.agent.n_m_states):                
                beta = self.beta_coefficient * self.agent.beta                
            else:
                beta = self.agent.beta / self.beta_coefficient
            self.agent.eta = self.agent.eta * sum / (sum-self.agent.beta+beta)
            self.agent.nu = self.agent.nu * sum / (sum-self.agent.beta+beta)
            self.agent.beta = beta * sum / (sum-self.agent.beta+beta)
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
    
    def load(self, common_path, specific_path, load_memory=True):
        self.agent.load(common_path, specific_path, load_memory=load_memory)

    
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
