import gym
import torch
import numpy as np

from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from nets_seq import (Memory, v_Net, q_Net, DQN, s_Net, c_Net, RND_Net, d_Net, HeapPriorityQueue)

import os
import time
import pickle
from sys import stdout
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import curses
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

###########################################################################
#
#                           General methods
#
###########################################################################
def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def scale_action(a, min, max):
    return (0.5*(a+1.0)*(max-min) + min)

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes).to(device) 
    return y[labels]

def set_seed(n_seed):
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda": torch.cuda.manual_seed(n_seed)

def is_float(x):
    return isinstance(x, float)

def is_tensor(x):
    return isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor)

def vectorized_multinomial(prob_matrix):
    items = np.arange(prob_matrix.shape[1])
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0],1)
    k = (s < r).sum(axis=1)
    return items[k]

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
    def __init__(self, s_dim, a_dim, n_tasks, params, seed=0):

        self.params = params.copy()
        default_params = {
                            'n_concepts': 10,
                            'decision_type': 'epsilon',
                            'alpha': {
                                        'sl': 1.0,
                                        'ql': 1.0,
                                        'cl': 1e-6
                                    },
                            'beta': {
                                        'H(S|s)': 1e-20,
                                        'H(S|R;T)': 1e-20,
                                        'H(nS|S,A,T)': 1e-20,
                                        'H(S|T)': 1e-20
                                    },
                            'init_epsilon': 0.00,
                            'min_epsilon': 0.00,
                            'delta_epsilon': 1.6e-6,
                            'epsilon_tl': 0.05,
                            'init_threshold_entropy_alpha': 0.0,
                            'init_threshold_entropy_alpha_cl': np.log(10),
                            'delta_threshold_entropy_alpha': 8e-6,
                            'delta_threshold_entropy_alpha_cl': 3.2e-6,
                            'min_threshold_entropy_alpha_ql': np.log(2),
                            'min_threshold_entropy_alpha_cl': np.log(2),
                            'DQN_learning_type': 'DQL',
                            'DQL_epsds_target_update': 6000,
                            # 'init_beta': 0.4,
                            # 'max_beta': 1.0,
                            # 'delta_beta': 3.0e-4/4.0,
                            
                            'lr': {
                                    'sl': {
                                            'q': 3e-4,
                                            'v': 3e-4,
                                            'pi': 3e-4,
                                            'alpha': 3e-4,
                                            'v_target': 5e-3
                                        },
                                    'ql': {
                                            'q': 3e-4,
                                            'v': 3e-4,
                                            'alpha': 3e-4,
                                            'v_target': 5e-3
                                        },
                                    'cl': {
                                            'alpha': 3e-2,
                                            'beta': 3e-2
                                        }
                                    },

                            'dim_excluded': {
                                        'init': 2,
                                        'middle': 7,
                                        'last': 60
                                    },
                            
                            'batch_size': {
                                            'sl': 256,
                                            'ql': 256
                                        },
                            'RND_factor': 1.0,
                            'memory_capacity': 400000,
                            'gamma_E': 0.99,
                            'gamma_I': 0.95,
                            'gamma_tl': 0.95,
                            'clip_value': 1.0,
                            'classification_with_entropies': True,
                            'n_update_cycles_ps': 10                                                                                     
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

        self.n_tasks = n_tasks
        self.seed = seed
        self.n_skills = n_tasks['sl']
        self.counter = {
                        'sl': 0,
                        'ql': 0
                    }

        self.n_concepts = self.params['n_concepts']
        self.dim_excluded = self.params['dim_excluded']
        self.batch_size = self.params['batch_size']
        self.lr = self.params['lr']
        self.gamma_E = self.params['gamma_E']
        self.gamma_I = self.params['gamma_I']
        self.gamma_tl = self.params['gamma_tl']
        self.clip_value = self.params['clip_value']
        self.decision_type = self.params['decision_type']
        self.DQN_learning_type = self.params['DQN_learning_type']
        self.DQL_epsds_target_update = self.params['DQL_epsds_target_update']
        self.RND_factor = self.params['RND_factor']
        self.classification_with_entropies = self.params['classification_with_entropies']
        self.n_update_cycles_ps = self.params['n_update_cycles_ps']

        # Metric weights
        self.min_threshold_entropy_alpha = {
                                            'sl': -a_dim*1.0,
                                            'ql': self.params['min_threshold_entropy_alpha_ql'],
                                            'cl': self.params['min_threshold_entropy_alpha_cl']
                                        }
        self.threshold_entropy_alpha = {
                                        'sl': self.params['init_threshold_entropy_alpha'],
                                        'ql': self.params['init_threshold_entropy_alpha'],
                                        'cl': self.params['init_threshold_entropy_alpha_cl']
                                    }
        self.delta_threshold_entropy_alpha = self.params['delta_threshold_entropy_alpha']
        self.delta_threshold_entropy_alpha_cl = self.params['delta_threshold_entropy_alpha_cl']
        alpha = self.params['alpha']
        self.alpha = {}
        self.alpha['cl'] = alpha['cl']
        for learning_type in ['sl', 'ql']:
            self.alpha[learning_type] = (alpha[learning_type] * torch.ones(self.n_tasks[learning_type]).float().to(device) if is_float(alpha[learning_type]) else 
                            (alpha[learning_type].float().to(device) if is_tensor(alpha[learning_type]) else torch.from_numpy(alpha[learning_type]).float().to(device)))
        beta = self.params['beta']
        self.beta = {'H(S|s)': beta['H(S|s)']}
        for entropy in ['H(S|R;T)', 'H(nS|S,A,T)', 'H(S|T)']:
            if entropy in ['H(S|R;T)', 'H(S|T)']:
                self.beta[entropy] = (beta[entropy] * torch.ones(self.n_tasks[learning_type]).float().to(device) if is_float(beta[entropy]) else 
                                (beta[entropy].float().to(device) if is_tensor(beta[entropy]) else torch.from_numpy(beta[entropy]).float().to(device)))
            else:
                self.beta[entropy] = (beta[entropy] * torch.ones(self.n_tasks[learning_type], self.n_concepts, self.n_skills+1).float().to(device) if is_float(beta[entropy]) else 
                                (beta[entropy].float().to(device) if is_tensor(beta[entropy]) else torch.from_numpy(beta[entropy]).float().to(device)))

        self.epsilon = self.params['init_epsilon']
        self.min_epsilon = self.params['min_epsilon']
        self.delta_epsilon = self.params['delta_epsilon']
        self.epsilon_tl = self.params['epsilon_tl']
        # self.beta = self.params['init_beta']
        # self.max_beta = self.params['max_beta']
        # self.delta_beta = self.params['delta_beta']
        
        # Nets and memory
        self.v = {
                            'sl': v_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), n_tasks['sl'], lr=self.lr['sl']['v']).to(device),
                            'ql': v_Net(s_dim-self.dim_excluded['middle'], n_tasks['ql'], lr=self.lr['ql']['v']).to(device),
                            'tl': np.random.rand(self.n_tasks['tl'], self.n_concepts, 1)
                        }
        self.v_target = {
                            'sl': v_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), n_tasks['sl'], lr=self.lr['sl']['v']).to(device),
                            'ql': v_Net(s_dim-self.dim_excluded['middle'], n_tasks['ql'], lr=self.lr['ql']['v']).to(device)
                        }
        self.critic1 = {
                            'sl': q_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, n_tasks['sl'], lr=self.lr['sl']['q']).to(device),
                            'ql': DQN(s_dim-self.dim_excluded['middle'], self.n_skills+1, n_tasks['ql'], lr=self.lr['ql']['q']).to(device),
                            'tl': self.v['tl'].repeat(self.n_skills+1, axis=2)
                        }
        self.critic2 = {
                            'sl': q_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, n_tasks['sl'], lr=self.lr['sl']['q']).to(device),
                            'ql': DQN(s_dim-self.dim_excluded['middle'], self.n_skills+1, n_tasks['ql'], lr=self.lr['ql']['q']).to(device),
                            'tl': self.v['tl'].repeat(self.n_skills+1, axis=2)
                        }
        
        self.actor = s_Net(self.n_skills, s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, lr=self.lr['sl']['pi']).to(device)
        self.classifier = c_Net(self.n_concepts, s_dim-self.dim_excluded['middle'], self.n_skills+1, n_tasks=self.n_tasks['ql'], lr=3e-5).to(device)

        self.memory = {
                        'sl':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'ql':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'tl':  Memory(self.params['memory_capacity'], n_seed=self.seed)
                    }
        
        self.counts = {
            'Ntsa': np.zeros([self.n_tasks['tl'], self.n_concepts, self.n_skills+1]).astype(int),
            'Ntsas': np.zeros([self.n_tasks['tl'], self.n_concepts, self.n_skills+1, self.n_concepts]).astype(int)
        }

        self.priority_queue = HeapPriorityQueue(self.n_tasks['tl'])
        
        # self.RNDnet = {
        #                 'sl': RND_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), self.n_tasks['sl']).to(device),
        #                 'ql': RND_Net(s_dim-self.dim_excluded['init'], self.n_tasks['ql']).to(device)
        #             }

        # self.density = d_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), self.n_tasks['sl']).to(device)
        # self.density_target = torch.ones(self.n_tasks['sl'], 8).float().to(device)/8.0
        # self.density = torch.ones(self.n_tasks['sl'], 8).float().to(device)/8.0
        # self.prior_weight = 8*1000

        updateNet(self.v_target['sl'], self.v['sl'],1.0)
        updateNet(self.critic2['ql'], self.critic1['ql'],1.0)    
    
    # def update_density(self, task, idx):
    #     self.density[task, idx] += 1.0/self.prior_weight
    #     self.density[task, :] /= 1.0+1.0/self.prior_weight

    # def update_target_density(self):
    #     self.density_target = self.density.clone()

    def memorize(self, event, learning_type, init=False):
        if init:
            self.memory[learning_type].store(event[np.newaxis,:])
        else:
            self.memory[learning_type].store(event.tolist())
    
    def relate_concept(self, state, explore=True):
        state_cuda = torch.FloatTensor(state[self.dim_excluded['middle']:]).to(device).view(1,-1)
        with torch.no_grad():
            return self.classifier.sample_concept(state_cuda, explore=explore)[0]

    def decide(self, state, task, learning_type, explore=True):
        if learning_type == 'ql':
            skill = self.decide_q_dist(state, task, explore=explore) if self.decision_type == 'q_dist' else self.decide_epsilon(state, task, explore=explore)
            return skill 
        elif learning_type == 'tl':
            concept = self.relate_concept(state, explore=explore)
            skill = self.decide_concept_skill_pair(concept, task, explore=explore)
            return concept, skill
    
    def decide_concept_skill_pair(self, concept, task, explore=True):
        q = self.critic1['tl'][task, concept, :]
        epsilon = 0.0 if not explore else self.epsilon_tl 
        skill = q.argmax() if np.random.rand() > epsilon else np.random.randint(self.n_skills+1)
        return skill       

    def decide_q_dist(self, state, task, explore=True):
        s_cuda = torch.FloatTensor(state[self.dim_excluded['middle']:]).to(device).view(1,-1)
        q = self.critic1['ql'](s_cuda).squeeze(0)[task,:] if np.random.rand() > 0.5 else self.critic2['ql'](s_cuda).squeeze(0)[task,:]
        with torch.no_grad():
            pi = torch.exp((q-q.max())/(self.alpha['ql'][task]+1e-6)).view(-1)
            pi = pi / pi.sum()
            skill = Categorical(probs=pi).sample().item() if explore else pi.argmax().item()
            return skill

    def decide_epsilon(self, state, task, explore=True):
        s_cuda = torch.FloatTensor(state[self.dim_excluded['middle']:]).to(device).view(1,-1)
        with torch.no_grad():
            # qe, qi = self.critic1['ql'](s_cuda)
            # qe, qi = qe.squeeze(0)[task,:], qi.squeeze(0)[task,:]
            qe = self.critic1['ql'](s_cuda)
            qe = qe.squeeze(0)[task,:]
            # epsilon = self.epsilon if explore else 0.0
            # skill = (qe+qi).argmax().item() if np.random.rand() > epsilon else np.random.randint(self.n_skills+1)
            skill = (qe).argmax().item() # if np.random.rand() > epsilon else np.random.randint(self.n_skills+1)
            return skill            

    def act(self, state, skill, explore=True):
        s_cuda = torch.FloatTensor(state[self.dim_excluded['init']:-self.dim_excluded['last']]).to(device)
        with torch.no_grad():
            a = self.actor.sample_action(s_cuda, skill, explore=explore) if skill < self.n_tasks['sl'] else np.zeros(self.a_dim)            
            return a

    def learn_DQN(self, only_metrics=False):
        if not only_metrics:
            self.learn_DQN_DQL() if self.DQN_learning_type == 'DQL' else self.learn_DQN_SAC(only_metrics=only_metrics)
        else:
            metrics = {} if self.DQN_learning_type == 'DQL' else self.learn_DQN_SAC(only_metrics=only_metrics)
            return metrics

    def learn_DQN_DQL(self):
        self.counter['ql'] += 1
        # batch, indices, priorities = self.memory['ql'].sample(self.batch_size['ql'])
        batch = self.memory['ql'].sample(self.batch_size['ql'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['middle']:self.s_dim]).to(device)
            A_batch = batch[:,self.s_dim].astype('int')
            r_batch = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.s_dim+2+self.dim_excluded['middle']:2*self.s_dim+2]).to(device)
            d_batch = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T_batch = batch[:,2*self.s_dim+3].astype('int')  

            # Optimize q networks
            # qe, qi = self.critic1['ql'](s_batch)
            # qe, qi = qe[np.arange(batch_size), T_batch, A_batch].view(-1,1), qi[np.arange(batch_size), T_batch, A_batch].view(-1,1)
            # qen, qin = self.critic1['ql'](ns_batch)
            # qen, qin = qen[np.arange(batch_size), T_batch, :], qin[np.arange(batch_size), T_batch, :]
            # qen_target, qin_target =  self.critic2['ql'](ns_batch)
            # qen_target, qin_target =  qen_target[np.arange(batch_size), T_batch, :], qin_target[np.arange(batch_size), T_batch, :]
            qe = self.critic1['ql'](s_batch)
            qe = qe[np.arange(batch_size), T_batch, A_batch].view(-1,1)
            qen = self.critic1['ql'](ns_batch)
            qen = qen[np.arange(batch_size), T_batch, :]
            qen_target =  self.critic2['ql'](ns_batch)
            qen_target =  qen_target[np.arange(batch_size), T_batch, :]

            # RDN_error = self.RNDnet(ns_batch)
            # best_skills = (qen+qin).argmax(1)
            best_skills = qen.argmax(1)
            qe_approx = r_batch/10.0 + self.gamma_E * qen_target[np.arange(batch_size), best_skills].view(-1,1) * (1.0-d_batch)
            # qi_approx = RDN_error.detach() + self.gamma_I * qin_target[np.arange(batch_size), best_skills].view(-1,1) * (1.0-d_batch)

            # new_sampling_priorities = list((((qe + qi - qe_approx - qi_approx).squeeze(1).abs()+1e-2)**0.6).detach().cpu().numpy())

            # IS_weights = (priorities.min()/priorities.view(-1,1))**self.beta
            q_loss = self.critic1['ql'].loss_func(qe, qe_approx.detach())# + self.critic1['ql'].loss_func(qi, qi_approx.detach()))*IS_weights
            self.critic1['ql'].optimizer.zero_grad()
            q_loss.mean().backward()
            clip_grad_norm_(self.critic1['ql'].parameters(), self.clip_value)
            self.critic1['ql'].optimizer.step()

            # self.RNDnet.optimizer.zero_grad()
            # (RDN_error*IS_weights).mean().backward()
            # clip_grad_norm_(self.RNDnet.predictor.parameters(), self.clip_value)
            # self.RNDnet.optimizer.step()
            
            # self.memory['ql'].update_weights(indices, new_sampling_priorities)

            if self.counter['ql'] % self.DQL_epsds_target_update == 0:
                updateNet(self.critic2['ql'], self.critic1['ql'], 1.0)
                self.counter['ql'] = 0

            # Anneal epsilon and beta
            self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])
            # self.beta = np.min([self.beta + self.delta_beta, self.max_beta])
    
    def learn_DQN_SAC(self, only_metrics=False):
        batch = self.memory['ql'].sample(self.batch_size['ql'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['middle']:self.s_dim]).to(device)
            A_batch = batch[:,self.s_dim].astype('int')
            r_batch = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.s_dim+2+self.dim_excluded['middle']:2*self.s_dim+2]).to(device)
            d_batch = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T_batch = batch[:,2*self.s_dim+3].astype('int')  

            # Optimize q networks
            q1 = self.critic1['ql'](s_batch)
            q2 = self.critic2['ql'](s_batch)
            q = torch.min(torch.stack([q1, q2]), 0)[0].detach()
            if not only_metrics:               
                q1_AT = q1[np.arange(batch_size), T_batch, A_batch]
                q2_AT = q2[np.arange(batch_size), T_batch, A_batch]

                next_v = self.v_target['ql'](ns_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q_approx = r_batch + self.gamma_E * next_v * (1-d_batch)
                
                q1_loss = self.critic1['ql'].loss_func(q1_AT, q_approx.detach())
                self.critic1['ql'].optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1['ql'].parameters(), self.clip_value)
                self.critic1['ql'].optimizer.step()
                
                q2_loss = self.critic2['ql'].loss_func(q2_AT, q_approx.detach())
                self.critic2['ql'].optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2['ql'].parameters(), self.clip_value)
                self.critic2['ql'].optimizer.step()                

            # Optimize v network
            pi = torch.exp((q - q.max(2, keepdim=True)[0])/ (self.alpha['ql'].view(1,-1,1)+1e-6))
            pi = pi / pi.sum(2, keepdim=True)
            q_mean = (pi * q).sum(2).detach()
            HA_given_sT = -(pi * torch.log(pi + 1e-10)).sum(2).detach()
            v_approx = (q_mean + self.alpha['ql'].view(1,-1) * HA_given_sT)[np.arange(batch_size), T_batch].view(-1,1) 

            if not only_metrics:
                v = self.v['ql'](s_batch)[np.arange(batch_size), T_batch].view(-1,1)
            
            task_mask = torch.zeros(batch_size, self.n_tasks['ql']).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            
            HA_s_given_T = (HA_given_sT * task_mask_distribution).sum(0)
            HA_sT = HA_s_given_T.mean()
            alpha_gradient = HA_s_given_T.detach() - self.threshold_entropy_alpha['ql']

            if not only_metrics:
                v_loss = ((v - v_approx.detach())**2).mean()
                self.v['ql'].optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v['ql'].parameters(), self.clip_value)
                self.v['ql'].optimizer.step()
                updateNet(self.v_target['ql'], self.v['ql'], self.lr['ql']['v_target'])

                # Optimize dual variable                
                log_alpha = torch.log(self.alpha['ql'] + 1e-6)
                log_alpha -= self.lr['ql']['alpha'] * alpha_gradient
                self.alpha['ql'] = torch.exp(log_alpha).clamp(1e-10, 1e+3)
                    
        else:
            HA_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(A|s,T)': HA_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics
    
    def learn_skills(self, only_metrics=False):
        # if not only_metrics: self.counter['sl'] += 1

        batch = self.memory['sl'].sample(self.batch_size['sl'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['init']:self.s_dim-self.dim_excluded['last']]).to(device)
            a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
            r_batch = torch.FloatTensor(batch[:,self.sa_dim]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.sa_dim+1+self.dim_excluded['init']:self.sars_dim-self.dim_excluded['last']]).to(device)
            d_batch = torch.FloatTensor(batch[:,self.sars_dim]).view(-1,1).to(device)
            T_batch = batch[:,self.sarsd_dim].astype('int')  
            # angle_batch = batch[:,self.sarsd_dim+1].astype('int')  

            if not only_metrics:
                # Optimize q networks
                q1_E = self.critic1['sl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q2_E = self.critic2['sl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                next_v_E = self.v_target['sl'](ns_batch)[np.arange(batch_size), T_batch].view(-1,1)
                # q1_E, q1_I = self.critic1['sl'](s_batch, a_batch)
                # q1_E, q1_I = q1_E[np.arange(batch_size), T_batch].view(-1,1), q1_I[np.arange(batch_size), T_batch].view(-1,1)
                # q2_E, q2_I = self.critic2['sl'](s_batch, a_batch)
                # q2_E, q2_I = q2_E[np.arange(batch_size), T_batch].view(-1,1), q2_I[np.arange(batch_size), T_batch].view(-1,1)
                # next_v_E, next_v_I = self.v_target['sl'](ns_batch)
                # next_v_E, next_v_I = next_v_E[np.arange(batch_size), T_batch].view(-1,1), next_v_I[np.arange(batch_size), T_batch].view(-1,1)
                # RDN_error = self.RNDnet['sl'](ns_batch.detach(), T_batch)
                # s_approx_batch, mu_latent, log_sigma_latent = self.density(s_batch, T_batch)
                # density_loss = self.density.loss_func(s_batch, s_approx_batch, mu_latent, log_sigma_latent)

                # angle_loss = np.log(1.0/8.0) - torch.log(self.density_target[T_batch, angle_batch]).view(-1,1)

                q_approx_E = r_batch + self.gamma_E * next_v_E * (1-d_batch)
                # q_approx_I = angle_loss + self.gamma_I * next_v_I * (1-d_batch)
                # q_approx_I = RDN_error.detach() + self.gamma_I * next_v_I * (1-d_batch)
                # q_approx_I = density_loss.detach() + self.gamma_I * next_v_I * (1-d_batch)
                
                q1_loss = self.critic1['sl'].loss_func(q1_E, q_approx_E.detach()) # + self.critic1['sl'].loss_func(q1_I, q_approx_I.detach())
                self.critic1['sl'].optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1['sl'].parameters(), self.clip_value)
                self.critic1['sl'].optimizer.step()
                
                q2_loss = self.critic2['sl'].loss_func(q2_E, q_approx_E.detach()) # + self.critic2['sl'].loss_func(q2_I, q_approx_I.detach())
                self.critic2['sl'].optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2['sl'].parameters(), self.clip_value)
                self.critic2['sl'].optimizer.step()

                # self.RNDnet['sl'].optimizer.zero_grad()
                # RDN_error.mean().backward()
                # clip_grad_norm_(self.RNDnet['sl'].predictor.parameters(), self.clip_value)
                # self.RNDnet['sl'].optimizer.step()

                # self.density.optimizer.zero_grad()
                # density_loss.mean().backward()
                # clip_grad_norm_(self.density.parameters(), self.clip_value)
                # self.density.optimizer.step()                

            # Optimize v network
            a_batch_A, log_pa_sApT_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch)
            a_batch = a_batch_A[np.arange(batch_size), T_batch, :]
            log_pa_sT = log_pa_sApT_A[np.arange(batch_size), :, T_batch]
            
            q1_off_E = self.critic1['sl'](s_batch.detach(), a_batch)
            q2_off_E = self.critic2['sl'](s_batch.detach(), a_batch)
            # q1_off_E, q1_off_I = self.critic1['sl'](s_batch.detach(), a_batch)
            # q2_off_E, q2_off_I = self.critic2['sl'](s_batch.detach(), a_batch)
            q_off_E = torch.min(torch.stack([q1_off_E, q2_off_E]), 0)[0]
            # q_off_I = torch.min(torch.stack([q1_off_I, q2_off_I]), 0)[0]
            
            v_approx_E = (q_off_E - self.alpha['sl'].view(1,-1) * log_pa_sT)[np.arange(batch_size), T_batch].view(-1,1)
            # v_approx_I = q_off_I[np.arange(batch_size), T_batch].view(-1,1) 

            if not only_metrics:
                v_E = self.v['sl'](s_batch)[np.arange(batch_size), T_batch].view(-1,1)
                # v_E, v_I = self.v['sl'](s_batch)
                # v_E, v_I = v_E[np.arange(batch_size), T_batch].view(-1,1), v_I[np.arange(batch_size), T_batch].view(-1,1) 
            
            task_mask = torch.zeros(batch_size, self.n_tasks['sl']).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            Ha_sT = -(log_pa_sT * task_mask_distribution).sum(0)
            alpha_gradient = Ha_sT.detach() - self.threshold_entropy_alpha['sl']

            if not only_metrics:
                v_loss = self.v['sl'].loss_func(v_E, v_approx_E.detach()) # + self.v['sl'].loss_func(v_I, v_approx_I.detach())
                self.v['sl'].optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v['sl'].parameters(), self.clip_value)
                self.v['sl'].optimizer.step()
                updateNet(self.v_target['sl'], self.v['sl'], self.lr['sl']['v_target'])

                # Optimize skill network
                pi_loss = -(v_approx_E).mean()# /self.RND_factor + v_approx_I).mean()
                self.actor.optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()

                # Optimize dual variable                
                log_alpha = torch.log(self.alpha['sl'] + 1e-6)
                log_alpha -= self.lr['sl']['alpha'] * alpha_gradient
                self.alpha['sl'] = torch.exp(log_alpha).clamp(1e-10, 1e+3)

                self.threshold_entropy_alpha['sl'] = np.max([self.threshold_entropy_alpha['sl'] - self.delta_threshold_entropy_alpha, self.min_threshold_entropy_alpha['sl']])

                # if self.counter['sl'] % (self.prior_weight * self.n_tasks['sl']) == 0:
                # self.update_target_density()
                # self.counter['sl'] = 0
                    
        else:
            log_pa_sT = torch.zeros(1).to(device)  
            Ha_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(a|s,T)': Ha_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics
    
    def learn_concepts(self):
        batch = self.memory['ql'].sample(self.batch_size['ql'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['middle']:self.s_dim]).to(device)
            A_batch = batch[:,self.s_dim].astype('int')
            r_batch = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1).to(device)  
            ns_batch = torch.FloatTensor(batch[:,self.s_dim+2+self.dim_excluded['middle']:2*self.s_dim+2]).to(device)
            T_batch = batch[:,2*self.s_dim+3].astype('int')

            q = self.critic1['ql'](s_batch)[np.arange(batch_size), T_batch, :]
            A_batch_off = q.argmax(1)

            A_batch_one_hot_off = torch.zeros(batch_size, self.n_skills+1).to(device)
            A_batch_one_hot_off[np.arange(batch_size), A_batch_off] = torch.ones(batch_size,).to(device)
            A_distribution_off = A_batch_one_hot_off.sum(0, keepdim=True)
            A_distribution_off /= A_distribution_off.sum()

            A_batch_one_hot = torch.zeros(batch_size, self.n_skills+1).to(device)
            A_batch_one_hot[np.arange(batch_size), A_batch] = torch.ones(batch_size,).to(device)

            T_batch_one_hot = torch.zeros(batch_size, self.n_tasks['cl']).to(device)
            T_batch_one_hot[np.arange(batch_size), T_batch] = torch.ones(batch_size,).to(device)

            A_predicted, PA_ST, log_PA_ST, S, PS_s, log_PS_s, z = self.classifier.sample_skills(s_batch, T_batch_one_hot)
            HS_s = -(PS_s * log_PS_s).sum(1).mean()

            PS_s_T = PS_s.unsqueeze(1) * T_batch_one_hot.unsqueeze(2)
            ps_ST = PS_s_T / (PS_s_T.sum(0, keepdim=True) + 1e-10)
            PS_T = PS_s_T.mean(0)
            PS_T = PS_T / (PS_T.sum(1, keepdim=True) + 1e-10)
            i_ST = Categorical(probs=ps_ST.transpose(0,1).transpose(1,2)).sample(sample_shape=torch.Size([batch_size//4]))
            sigma = (r_batch.max() - r_batch.min()) / batch_size
            R_iST = r_batch[i_ST] + sigma * torch.randn(batch_size//4, self.n_tasks['cl'], self.n_concepts).to(device)
            log_pR_k_iST = -0.5 * (torch.log(2*np.pi*sigma**2+1e-10).view(1,1,1,1) + (R_iST.unsqueeze(1) - r_batch.view(1,-1,1,1))**2/sigma**2)
            log_pR_pST_iS = torch.logsumexp(log_pR_k_iST.unsqueeze(4) + torch.log(ps_ST+1e-10).unsqueeze(0).unsqueeze(3), dim=1)
            log_PR_ST_i = log_pR_pST_iS[:, :, np.arange(self.n_concepts), np.arange(self.n_concepts)]
            log_PR_T_iS = torch.logsumexp(log_pR_pST_iS + torch.log(PS_T+1e-10).unsqueeze(0).unsqueeze(2), dim=3)
            log_PS_RT = log_PR_ST_i + torch.log(PS_T+1e-10).unsqueeze(0) - log_PR_T_iS
            HS_R_T = - (PS_T * log_PS_RT.mean(0)).sum(1, keepdim=True)

            nS, PnS_ns, log_PnS_ns = self.classifier.sample_concepts(ns_batch)
            AT_mask = T_batch_one_hot.unsqueeze(2) * A_batch_one_hot.unsqueeze(1)
            PnS_S_iT = (PS_s.unsqueeze(2) * PnS_ns.unsqueeze(1)).unsqueeze(1) / (PS_T.unsqueeze(0).unsqueeze(3) + 1e-10)
            PnS_SAT = torch.einsum('ihjl,ihk->hjkl', PnS_S_iT, AT_mask)
            # N = PnS_SAT.sum(3, keepdim=True)
            # N = N * (N.detach() > 0).float().to(device) + torch.ones_like(N.detach()) * (1 - (N.detach() > 0)).float().to(device)
            PnS_SAT = PnS_SAT / (PnS_SAT.sum(3, keepdim=True) + 1e-10)
            assert torch.all(PnS_SAT == PnS_SAT), 'EXPLOSION nS!'
            HnS_SAT = -(PnS_SAT * torch.log(PnS_SAT + 1e-10)).sum(3) 

            HS_T = -(PS_T * torch.log(PS_T + 1e-10)).sum(1, keepdim=True)
            AT_mask_off = T_batch_one_hot.unsqueeze(2) * A_batch_one_hot_off.unsqueeze(1)
            PA_T = AT_mask_off.sum(0)
            # NA_T = PA_T.sum(1, keepdim=True)
            # NA_T = NA_T * (NA_T.detach() > 0).float().to(device) + torch.ones_like(NA_T.detach()) * (1 - (NA_T.detach() > 0)).float().to(device)
            PA_T = PA_T / (PA_T.sum(1, keepdim=True) + 1e-10)
            assert torch.all(PA_T == PA_T), 'EXPLOSION A!'
            HA_T = -(PA_T * torch.log(PA_T + 1e-10)).sum(1, keepdim=True)
            
            beta_Ss_gradient = (HS_s - self.threshold_entropy_alpha['cl']).detach().item()
            beta_gradient_SR = (HS_R_T.squeeze(1) - self.threshold_entropy_alpha['cl']).detach()
            beta_gradient_nSSAT = (HnS_SAT - self.threshold_entropy_alpha['cl']).detach()
            beta_gradient_ST = (HA_T - HS_T).squeeze(1).detach()
            classification_loss = (-log_PA_ST / ((self.n_skills + 1) * A_distribution_off.clamp(1.0/batch_size, 1.0)))[np.arange(batch_size), A_batch_off].mean()

            classifier_loss = 1e+3*classification_loss
            if self.classification_with_entropies:
                classifier_loss += self.beta['H(S|s)'] * HS_s
                classifier_loss += (self.beta['H(S|R;T)'].view(-1,1) * HS_R_T).mean()
                classifier_loss += (self.beta['H(nS|S,A,T)'] * HnS_SAT).mean()
                classifier_loss -= (self.beta['H(S|T)'].view(-1,1) * HS_T).mean()

            self.classifier.optimizer.zero_grad()
            classifier_loss.backward(retain_graph=True)
            clip_grad_norm_(self.classifier.parameters(), self.clip_value)
            self.classifier.optimizer.step()
            assert torch.all(self.classifier.lv1e.weight==self.classifier.lv1e.weight), 'EXPLOSION c'

            # Optimize dual variables  
            log_beta_Ss = np.log(self.beta['H(S|s)'])
            log_beta_Ss += self.lr['cl']['beta'] * beta_Ss_gradient
            self.beta['H(S|s)'] = np.exp(log_beta_Ss).clip(1e-20, 1e+2)

            log_beta_SR = torch.log(self.beta['H(S|R;T)'])
            log_beta_SR += self.lr['cl']['beta'] * beta_gradient_SR
            self.beta['H(S|R;T)'] = torch.exp(log_beta_SR).clamp(1e-20, 1e+2)

            log_beta_nSSAT = torch.log(self.beta['H(nS|S,A,T)'])
            log_beta_nSSAT += self.lr['cl']['beta'] * beta_gradient_nSSAT
            self.beta['H(nS|S,A,T)'] = torch.exp(log_beta_nSSAT).clamp(1e-20, 1e+2)

            log_beta_ST = torch.log(self.beta['H(S|T)'])
            log_beta_ST += self.lr['cl']['beta'] * beta_gradient_ST
            self.beta['H(S|T)'] = torch.exp(log_beta_ST).clamp(1e-20, 1e+2)

            self.threshold_entropy_alpha['cl'] = np.max([self.threshold_entropy_alpha['cl'] - self.delta_threshold_entropy_alpha_cl, self.min_threshold_entropy_alpha['cl']])
        
            return(classification_loss.detach().item(), HS_s.detach().item(), HS_R_T.detach().mean().item(), 
                    HnS_SAT.detach().mean().item(), (HS_T-HA_T).detach().mean().item()) 
        else:
            print("Here")

    def prioritize(self, t, s, a):
        priority = np.abs(self.critic1['tl'][t,s,a] - self.critic2['tl'][t,s,a])
        self.priority_queue.add(t, s, 1.0/(priority+1e-6))
    
    def small_backup(self, t, s, a, ns, delta_V):
        self.critic1['tl'][t,s,a] += self.gamma_tl * self.counts['Ntsas'][t,s,a,ns]/self.counts['Ntsa'][t,s,a] * delta_V
    
    def state_backup(self, t, s):
        V = self.v['tl'][t,s]
        self.v['tl'][t,s] = self.critic1['tl'][t,s,:].max()
        delta_V = self.v['tl'][t,s] - V
        return delta_V

    def tabular_learning(self, sample):
        s, a, r, ns, _, t = sample
        self.update_tabular_model(s, a, r, ns, t)
        self.prioritize(t, s, a)
        self.prioritized_sweeping()
    
    def update_tabular_model(self, s, a, r, ns, t):
        self.counts['Ntsa'][t,s,a] += 1
        self.counts['Ntsas'][t,s,a,ns] += 1
        self.critic1['tl'][t,s,a] = (self.critic1['tl'][t,s,a] * float(self.counts['Ntsa'][t,s,a]-1) + r + self.gamma_tl*self.v['tl'][t,ns]) / float(self.counts['Ntsa'][t,s,a])
    
    def prioritized_sweeping(self):
        for cycle in range(0, self.n_update_cycles_ps):
            if not self.priority_queue.empty:
                t, s = self.priority_queue.pop()
                self.critic2['tl'][t,s,:] = self.critic1['tl'][t,s,:].copy()
                delta_V = self.state_backup(t, s)
                for ps, pa in itertools.product(np.arange(0,self.n_concepts), np.arange(0,self.n_skills+1)):
                    if self.counts['Ntsas'][t,ps,pa,s] > 0:
                        self.small_backup(t, ps, pa, s, delta_V)
                        self.prioritize(t, ps, pa)
    
    def estimate_metrics(self, learning_type):
        metrics = {}
        with torch.no_grad():
            if learning_type == 'sl':
                metrics = self.learn_skills(only_metrics=True)
            elif learning_type == 'ql':
                metrics = self.learn_DQN(only_metrics=True)
        return metrics
    
    def save(self, common_path, specific_path, learning_type):
        self.params['alpha'] = self.alpha
        self.params['beta'] = self.beta
        self.params['init_threshold_entropy_alpha'] = self.threshold_entropy_alpha['sl']
        self.params['init_threshold_entropy_alpha_cl'] = self.threshold_entropy_alpha['cl']
        self.params['init_epsilon'] = self.epsilon
        # self.params['init_density_C'] = self.density.C
        
        pickle.dump(self.params,open(common_path+'/agent_params.p','wb'))

        if learning_type in ['sl', 'ql']:
            data_batches = {'l': len(self.memory[learning_type].data)//20000+1}
            for i in range(0, data_batches['l']):
                if i+1 < data_batches['l']:
                    pickle.dump(self.memory[learning_type].data[20000*i:20000*(i+1)],open(common_path+'/memory_'+learning_type+str(i+1)+'.p','wb'))
                else:
                    pickle.dump(self.memory[learning_type].data[20000*i:-1],open(common_path+'/memory_'+learning_type+str(i+1)+'.p','wb'))
            pickle.dump(data_batches,open(common_path+'/data_batches_'+learning_type+'.p','wb'))

            torch.save(self.critic1[learning_type].state_dict(), specific_path+'_critic1_'+learning_type+'.pt')
            torch.save(self.critic2[learning_type].state_dict(), specific_path+'_critic2_'+learning_type+'.pt')
            torch.save(self.v[learning_type].state_dict(), specific_path+'_v_'+learning_type+'.pt')
            torch.save(self.v_target[learning_type].state_dict(), specific_path+'_v_target_'+learning_type+'.pt')
            # torch.save(self.RNDnet[learning_type].state_dict(), specific_path+'_RDN_'+learning_type+'.pt')
            
            if learning_type == 'sl':
                torch.save(self.actor.state_dict(), specific_path+'_actor.pt')
                # torch.save(self.density.state_dict(), specific_path+'_density.pt')
                # pickle.dump(self.density,open(specific_path+'_density.p','wb'))
                # pickle.dump(self.density_target,open(specific_path+'_density_target.p','wb'))
        elif learning_type == 'cl':
            suffix = '_we' if self.classification_with_entropies else '_woe'
            torch.save(self.classifier.state_dict(), specific_path+'_classifier' + suffix + '.pt')
    
    def load(self, common_path, specific_path, learning_type, load_memory=True):
        if learning_type in ['sl', 'ql']:
            if load_memory: 
                data_batches = pickle.load(open(common_path+'/data_batches_'+learning_type+'.p','rb'))
                pointer = 0
                for i in range(0, data_batches['l']):
                    data = pickle.load(open(common_path+'/memory_'+learning_type+str(i+1)+'.p','rb'))
                    self.memory[learning_type].data += data
                    pointer += len(data)
                self.memory[learning_type].pointer = pointer % self.memory[learning_type].capacity

            # self.density = pickle.load(open(specific_path+'_density.p','rb')).to(device)
            # self.density_target = pickle.load(open(specific_path+'_density_target.p','rb')).to(device)

            self.actor.load_state_dict(torch.load(specific_path+'_actor.pt'))
            # self.density.load_state_dict(torch.load(specific_path+'_density.pt'))
            self.actor.eval()
            # self.density.eval()
            # self.density.C = self.params['init_density_C']

            self.critic1[learning_type].load_state_dict(torch.load(specific_path+'_critic1_'+learning_type+'.pt'))
            self.critic2[learning_type].load_state_dict(torch.load(specific_path+'_critic2_'+learning_type+'.pt'))
            self.v[learning_type].load_state_dict(torch.load(specific_path+'_v_'+learning_type+'.pt'))
            self.v_target[learning_type].load_state_dict(torch.load(specific_path+'_v_target_'+learning_type+'.pt'))
            # self.RNDnet[learning_type].load_state_dict(torch.load(specific_path+'_RDN_'+learning_type+'.pt'))        

            self.critic1[learning_type].eval()
            self.critic2[learning_type].eval()
            self.v[learning_type].eval()
            self.v_target[learning_type].eval()
            # self.RNDnet[learning_type].eval()
        elif learning_type == 'cl':
            suffix = '_we' if self.classification_with_entropies else '_woe'
            self.classifier.load_state_dict(torch.load(specific_path+'_classifier' + suffix + '.pt'))
            self.classifier.train()
        

#----------------------------------------------
#
#                  System class
#
#----------------------------------------------
class System:
    def __init__(self, params, agent_params={}, skill_learning=True):
        
        self.params = params
        default_params = {
                            'seed': 1000,
                            'env_names_sl': [],
                            'env_names_ql': [],
                            'env_names_tl': [],
                            'env_steps_sl': 1,
                            'env_steps_ql': 10,
                            'env_steps_tl': 1000, 
                            'grad_steps': 1, 
                            'init_steps': 10000,
                            'max_episode_steps': 1000,
                            'tr_steps_sl': 1000,
                            'tr_steps_ql': 300,
                            'tr_epsd_sl': 1560,
                            'tr_epsd_ql': 1130,
                            'tr_epsd_tl': 1000,
                            'tr_steps_cl': 100000,
                            'tr_steps_tl': 100,
                            'eval_epsd_sl': 2,
                            'eval_epsd_interval': 10,
                            'eval_epsd_ql': 2,
                            'eval_epsd_tl': 10,
                            'batch_size': 256, 
                            'render': True, 
                            'reset_when_done': True, 
                            'store_video': False,
                            'storing_path': ''                           
                        }

        for key, value in default_params.items():
            if key not in self.params.keys():
                self.params[key] = value

        self.seed = self.params['seed']
        set_seed(self.seed)
        self.env_names = {
                            'sl': self.params['env_names_sl'],
                            'ql': self.params['env_names_ql'],
                            'cl': self.params['env_names_ql'],
                            'tl': self.params['env_names_tl']
                        }
        self.n_tasks = {
                            'sl': len(self.env_names['sl']),
                            'ql': len(self.env_names['ql']),
                            'cl': len(self.env_names['ql']),
                            'tl': len(self.env_names['tl'])
                        }
        self.steps = {
                        'env': {
                                'sl': self.params['env_steps_sl'],
                                'ql': self.params['env_steps_ql'],
                                'tl': self.params['env_steps_tl']
                            },
                        'grad': self.params['grad_steps'],

                        'init': self.params['init_steps'],
                        'tr': {
                                'sl': self.params['tr_steps_sl'],
                                'ql': self.params['tr_steps_ql'],
                                'cl': self.params['tr_steps_cl'],
                                'tl': self.params['tr_steps_tl']
                            }
                    }
        self.epsds = {
            'tr': {
                'sl': self.params['tr_epsd_sl'],
                'ql': self.params['tr_epsd_ql'],
                'tl': self.params['tr_epsd_tl']
            },
            'eval': {
                'sl': self.params['eval_epsd_sl'],
                'ql': self.params['eval_epsd_ql'],
                'tl': self.params['eval_epsd_tl'],
                'interval': self.params['eval_epsd_interval']
            },
        }
       
        self.batch_size = self.params['batch_size']
        self.render = self.params['render']
        self.store_video = self.params['store_video']
        self.reset_when_done = self.params['reset_when_done']
        self._max_episode_steps = self.params['max_episode_steps']
        
        self.envs = {}
        self.learning_type = 'sl' # if skill_learning else 'ql'

        self.set_envs()

        self.s_dim = self.envs[self.learning_type][0].observation_space.shape[0]
        self.a_dim = self.envs[self.learning_type][0].action_space.shape[0]        
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = self.s_dim*2 + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 2
        self.epsd_counter = 0
        self.task = 0

        self.min_action = self.envs[self.learning_type][0].action_space.low[0]
        self.max_action = self.envs[self.learning_type][0].action_space.high[0]

        self.agent = Agent(self.s_dim, self.a_dim, self.n_tasks, agent_params, seed=self.seed) 

    def set_envs(self):
        self.envs[self.learning_type] = []        
        for i in range(0, self.n_tasks[self.learning_type]):                    
            self.envs[self.learning_type].append(gym.make(self.env_names[self.learning_type][i]).unwrapped)
            print("Created env "+self.env_names[self.learning_type][i])
            self.envs[self.learning_type][i].reset()
            self.envs[self.learning_type][i].seed(self.seed)        
            self.envs[self.learning_type][i]._max_episode_steps = self._max_episode_steps
            self.envs[self.learning_type][i].rgb_rendering_tracking = True
    
    def reset(self, change_env=False):
        # self.envs[self.learning_type][self.task].close()        
        if change_env: self.task = (self.task+1) % self.n_tasks[self.learning_type]
        self.envs[self.learning_type][self.task].reset()        
    
    def get_obs(self):
        state = self.envs[self.learning_type][self.task]._get_obs().copy()
        return state
     
    def initialization(self, init_steps=0):         
        self.reset()
        if init_steps == 0: init_steps = self.steps['init']
        for init_step in range(0, init_steps * self.n_tasks[self.learning_type]):
            done = self.interaction_init()
            limit_reached = (init_step+1) % init_steps == 0
            if done or limit_reached: self.reset(change_env=limit_reached)
            if self.render: self.envs[self.learning_type][self.task].render()                        
        print("Finished initialization...")

    def interaction_init(self):  
        event = np.empty(self.t_dim)
        state = self.get_obs()
        action = 2.0*np.random.rand(self.a_dim)-1.0
        next_state, reward, done, info = self.envs[self.learning_type][self.task].step(action)  
        done = done and self.reset_when_done
        
        event[:self.s_dim] = state
        event[self.s_dim:self.sa_dim] = action
        event[self.sa_dim] = reward
        event[self.sa_dim+1:self.sars_dim] = next_state
        event[self.sars_dim] = float(done)
        event[self.sarsd_dim] = self.task
        event[self.sarsd_dim+1] = (info['angle']/(2*np.pi) * 8).astype(int)

        self.agent.memorize(event, self.learning_type)   
        return done

    def interaction(self, remember=True, explore=True, learn=True):  
        event = np.empty(self.t_dim)
        initial_state = self.get_obs()
        state = initial_state.copy()
        final_state = initial_state.copy()
        total_reward = 0.0
        done = end_step = False

        if self.learning_type == 'sl':
            skill = concept = self.task
        elif self.learning_type == 'ql':            
            skill = concept = self.agent.decide(state, self.task, self.learning_type, explore=explore)
        elif self.learning_type == 'tl':
            concept, skill = self.agent.decide(state, self.task, self.learning_type, explore=explore)

        max_env_step = self.steps['env'][self.learning_type]
        next_concept = concept

        for env_step in itertools.count(0):
            action = self.agent.act(state, skill, explore=explore if self.learning_type == 'sl' else False)
            scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
            next_state, reward, done, info = self.envs[self.learning_type][self.task].step(scaled_action)
            end_step = end_step or (done and self.reset_when_done)
            total_reward += reward
            final_state = np.copy(next_state)
            angle_idx = (info['angle'] * 8).astype(int) if self.learning_type == 'sl' else 0
            next_concept = self.agent.relate_concept(next_state, explore=explore)

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.sars_dim] = next_state
            event[self.sars_dim] = float(done)
            event[self.sarsd_dim] = self.task
            event[self.sarsd_dim+1] = angle_idx
        
            if remember and self.learning_type == 'sl': self.agent.memorize(event.copy(), self.learning_type)
            # if remember and learn and self.learning_type == 'sl': self.agent.update_density(self.task, angle_idx)
            if env_step < self.steps['env'][self.learning_type]-1: state = np.copy(next_state)
            if end_step or ((env_step+1) >= max_env_step) or (self.learning_type == 'tl' and next_concept != concept): break
            
        if self.learning_type == 'ql':
            event = np.empty(2*self.s_dim+4)
            event[:self.s_dim] = initial_state 
            event[self.s_dim] = skill
            event[self.s_dim+1] = total_reward
            event[self.s_dim+2:2*self.s_dim+2] = final_state
            event[2*self.s_dim+2] = float(done)
            event[2*self.s_dim+3] = self.task
            if remember: self.agent.memorize(event.copy(), self.learning_type)

        elif self.learning_type == 'tl':
            event = np.empty(6)
            event[0] = concept 
            event[1] = skill
            event[2] = total_reward
            event[3] = next_concept
            event[4] = float(done)
            event[5] = self.task        

        if learn:
            if self.learning_type == 'sl':
                for _ in range(0, self.steps['grad']):
                    self.agent.learn_skills()
            elif self.learning_type == 'ql':
                for _ in range(0, self.steps['grad']):
                    self.agent.learn_DQN()
            elif self.learning_type == 'tl':
                self.agent.tabular_learning(event)

        return total_reward, done, event

    def train_agent(self, initialization=True, skill_learning=True, storing_path='', rewards=[], metrics=[], losses=[], entropies=[], iter_0=0, q_learning=True, concept_learning=True, tabular_learning=True):
        if len(storing_path) == 0: storing_path = self.params['storing_path']

        if initialization:
            self.initialization()
            specific_path = storing_path + '/' + str(0)
            self.save(storing_path, specific_path)
        
        init_iter = iter_0
        if skill_learning:
            self.train_agent_skills(storing_path=storing_path, rewards=rewards, metrics=metrics, iter_0=init_iter)
            init_iter = 0
        
        self.learning_type = 'ql'        
        self.agent.memory['sl'].forget()

        if q_learning:
            self.set_envs()
            self.train_agent_skills(storing_path=storing_path, iter_0=init_iter)
            init_iter = 0

        self.learning_type = 'cl'
        if concept_learning:
            self.train_agent_concepts(storing_path=storing_path, iter_0=init_iter, losses=losses, entropies=entropies)
            init_iter = 0
        
        if tabular_learning:
            self.agent.memory['ql'].forget()
            self.learning_type = 'tl'
            self.set_envs()
            self.train_agent_skills(storing_path=storing_path, iter_0=init_iter)
    
    def train_agent_skills(self, iter_0=0, rewards=[], metrics=[], storing_path=''):        
        if self.render: self.envs[self.learning_type][self.task].render()         

        for epsd in range(0, self.epsds['tr'][self.learning_type]):
            change_env = False if epsd == 0 else True
            self.reset(change_env=change_env)
            iter_ = iter_0 + (epsd+1) // self.epsds['eval']['interval']
            
            for epsd_step in range(0, self.steps['tr'][self.learning_type]):
                if self.agent.memory[self.learning_type].len_data < self.batch_size:
                    done = self.interaction(learn=False)[1]
                else:
                    done = self.interaction(learn=True)[1]

                if self.render: self.envs[self.learning_type][self.task].render()

                if done: self.reset(change_env=False)

            if (epsd+1) % self.epsds['eval']['interval'] == 0:                
                r, _, m = self.eval_agent_skills(explore=False, iter_=iter_)[:3]
                metrics.append(m)
                rewards.append(r)
                np.savetxt(storing_path + '/metrics_'+self.learning_type+'.txt', np.array(metrics))               
                
                specific_path = storing_path + '/' + str(iter_)
                self.save(storing_path, specific_path=specific_path)
                np.savetxt(storing_path + '/mean_rewards_'+self.learning_type+'.txt', np.array(rewards))
    
    def train_agent_concepts(self, losses=[], entropies=[], storing_path='', iter_0=0):
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()

        suffix = '_we' if self.agent.classification_with_entropies else '_woe'

        for grad_step in range(0, self.steps['tr'][self.learning_type]):
            loss, HS_s, HS_RT, HnS_SAT, delta_H = self.agent.learn_concepts()
            losses.append(loss)
            entropies.append([HS_s, HS_RT, HnS_SAT, delta_H])

            stdscr.addstr(0, 0, "Iteration: {}".format(grad_step))
            stdscr.addstr(1, 0, "Classification Loss: {}".format(np.round(loss, 4)))
            stdscr.addstr(2, 0, "Entropy H(S|s): {}".format(np.round(HS_s,4)))
            stdscr.addstr(3, 0, "Entropy H(S|R;T): {}".format(np.round(HS_RT,4)))
            stdscr.addstr(4, 0, "Entropy H(nS|S,A,T): {}".format(np.round(HnS_SAT,4)))
            stdscr.addstr(5, 0, "Entropy diff. H(S|T)-H(A|T): {}".format(np.round(delta_H,4)))
            stdscr.refresh()

            if (grad_step + 1) % 50000 == 0:
                self.save(storing_path, storing_path+ '/' + str(iter_0+grad_step+1))
                np.savetxt(storing_path + '/concept_training_losses' + suffix + '.txt', np.array(losses))
                np.savetxt(storing_path + '/concept_training_entropies' + suffix + '.txt', np.array(entropies)) 

        curses.echo()
        curses.nocbreak()
        curses.endwin()
        
        # self.save(storing_path, storing_path+ '/' + str(iter_0+self.steps['tr'][self.learning_type]))
        # np.savetxt(storing_path + '/concept_training_losses' + suffix + '.txt', np.array(losses))
        # np.savetxt(storing_path + '/concept_training_entropies' + suffix + '.txt', np.array(entropies))            

    @property
    def entropy_metric(self):
        return self.learning_type == 'sl' or self.agent.DQN_learning_type == 'SAC'

    def eval_agent_skills(self, eval_epsds=0, explore=False, iter_=0, start_render=False, print_space=True, specific_path='video', max_step=0):   
        task = self.task
        self.task = 0
        self.reset()

        if start_render: self.envs[self.learning_type][self.task].render()
        if eval_epsds == 0: eval_epsds = self.epsds['eval'][self.learning_type] * self.n_tasks[self.learning_type]
        
        events = []
        rewards = []
        epsd_lenghts = []
        min_epsd_reward = 1.0e6
        max_epsd_reward = -1.0e6

        if self.entropy_metric:
            Ha_sT = []
            Ha_sT_average = 0.0
            entropy = 'H(a|s,T)' if self.learning_type == 'sl' else 'H(A|s,T)'
        
        for epsd in range(0, eval_epsds):

            if self.store_video: video = VideoWriter(specific_path + '_' + str(self.task) + '_' + str(epsd) + '_' + self.learning_type + '.avi', fourcc, float(FPS), (width, height))

            change_env = False if epsd == 0 else True
            self.reset(change_env=change_env)            
            if max_step <= 0: max_step = self.steps['tr'][self.learning_type]
            epsd_reward = 0.0

            for eval_step in itertools.count(0):            
                reward, done, event = self.interaction(explore=explore, learn=False)
                if self.learning_type == 'sl':
                    event[self.sa_dim] = reward  
                epsd_reward += reward              

                if self.store_video:
                    img = self.envs[self.learning_type][self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif self.render:
                    self.envs[self.learning_type][self.task].render()

                events.append(event)

                if done or ((eval_step + 1) >= max_step):
                    epsd_lenghts.append(eval_step + 1)
                    break

            metrics = self.agent.estimate_metrics(self.learning_type)
            if self.entropy_metric:                            
                Ha_sT.append(metrics[entropy])
                Ha_sT_average += (Ha_sT[-1] - Ha_sT_average)/(epsd+1)

            rewards.append(epsd_reward)
            min_epsd_reward = np.min([epsd_reward, min_epsd_reward])
            max_epsd_reward = np.max([epsd_reward, max_epsd_reward])
            average_reward = np.array(rewards).mean()
            
            if self.entropy_metric: 
                stdout.write("Iter %i, epsd %i, %s: %.4f, min r: %i, max r: %i, mean r: %i, epsd r: %i\r " %
                    (iter_, (epsd+1), entropy, Ha_sT_average, min_epsd_reward//1, max_epsd_reward//1, average_reward//1, epsd_reward//1))
            else:
                stdout.write("Iter %i, epsd %i, min r: %i, max r: %i, mean r: %i, epsd r: %i\r " %
                    (iter_, (epsd+1), min_epsd_reward//1, max_epsd_reward//1, average_reward//1, epsd_reward//1))
            stdout.flush()         

        if print_space: print("")

        if self.store_video: video.release()
        metric_vector = np.array([Ha_sT_average]) if self.entropy_metric else np.array([]) 
        
        self.task = task
        return rewards, np.array(events), metric_vector, np.array(epsd_lenghts)      
    
    def save(self, common_path, specific_path=''):
        self.params['learning_type'] = self.learning_type
        pickle.dump(self.params, open(common_path+'/params.p','wb'))
        self.agent.save(common_path, specific_path, self.learning_type)
    
    def load(self, common_path, iter_0_sl=0, iter_0_ql=0, iter_0_cl=0, load_memory=True):
        if iter_0_sl > 0:
            self.agent.load(common_path, common_path + '/' + str(iter_0_sl), 'sl', load_memory=(load_memory and iter_0_ql==0))
        if iter_0_ql > 0:
            self.agent.load(common_path, common_path + '/' + str(iter_0_ql), 'ql', load_memory=load_memory)
        if iter_0_cl > 0:
            self.agent.load(common_path, common_path + '/' + str(iter_0_cl), 'cl')

