import gym
import torch
import numpy as np

from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from nets_seq import (Memory, v_Net, q_Net, DQN, s_Net, c_Net)

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
def report_learning_progress(stdscr, iter_, transition_loss, reconstruction_error_wn, reconstruction_error_nn, reward_error_wn, reward_error_nn, max_error_factor, vq_loss):
    """Transition model learning progress"""
    stdscr.addstr(0, 0, "Iteration: {}".format(iter_))
    stdscr.addstr(1, 0, "Transition loss: {}".format(transition_loss))
    stdscr.addstr(2, 0, "Reconstruction loss (wn): {}".format(reconstruction_error_wn))
    stdscr.addstr(3, 0, "Reconstruction loss (nn): {}".format(reconstruction_error_nn))
    stdscr.addstr(4, 0, "Reward loss (wn): {}".format(reward_error_wn))
    stdscr.addstr(5, 0, "Reward loss (nn): {}".format(reward_error_nn))
    stdscr.addstr(6, 0, "Max error factor: {}".format(max_error_factor))
    stdscr.addstr(7, 0, "VQ loss: {}".format(vq_loss))
    stdscr.refresh()

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
                                        'ql': 1.0
                                    },
                            'init_epsilon': 1.0,
                            'min_epsilon': 0.1,
                            'delta_epsilon': 1.6e-6,
                            'init_threshold_entropy_alpha': 0.0,
                            'delta_threshold_entropy_alpha': 1.6e-5,
                            'min_threshold_entropy_alpha_ql': np.log(2),
                            'DQN_learning_type': 'DQL',

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
                                        }
                                    },

                            'dim_excluded': {
                                        'init': 2,
                                        'last': 60
                                    },
                            
                            'batch_size': {
                                            'sl': 256,
                                            'ql': 256
                                        },

                            'memory_capacity': 400000,
                            'gamma': 0.99,
                            'clip_value': 1.0                                                                                     
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

        self.n_concepts = self.params['n_concepts']
        self.dim_excluded = self.params['dim_excluded']
        self.batch_size = self.params['batch_size']
        self.lr = self.params['lr']
        self.gamma = self.params['gamma']
        self.clip_value = self.params['clip_value']
        self.decision_type = self.params['decision_type']
        self.DQN_learning_type = self.params['DQN_learning_type']

        # Metric weights
        self.min_threshold_entropy_alpha = {
                                            'sl': -a_dim*1.0,
                                            'ql': self.params['min_threshold_entropy_alpha_ql']
                                        }
        self.threshold_entropy_alpha = {
                                        'sl': self.params['init_threshold_entropy_alpha'],
                                        'ql': self.params['min_threshold_entropy_alpha_ql']
                                    }
        self.delta_threshold_entropy_alpha = self.params['delta_threshold_entropy_alpha']
        alpha = self.params['alpha']
        self.alpha = {}
        self.alpha['sl'] = (alpha['sl'] * torch.ones(self.n_tasks['sl']).float().to(device) if is_float(alpha['sl']) else 
                        (alpha['sl'].float().to(device) if is_tensor(alpha['sl']) else torch.from_numpy(alpha['sl']).float().to(device)))
        self.alpha['ql'] = (alpha['ql'] * torch.ones(self.n_tasks['ql']).float().to(device) if is_float(alpha['ql']) else 
                        (alpha['ql'].float().to(device) if is_tensor(alpha['ql']) else torch.from_numpy(alpha['ql']).float().to(device)))
        self.epsilon = self.params['init_epsilon']
        self.min_epsilon = self.params['min_epsilon']
        self.delta_epsilon = self.params['delta_epsilon']
        
        # Nets and memory
        self.critic1 = {
                            'sl': q_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, n_tasks['sl'], lr=self.lr['sl']['q']).to(device),
                            'ql': DQN(s_dim-self.dim_excluded['init'], self.n_skills+1, n_tasks['ql'], lr=self.lr['ql']['q']).to(device)
                        }
        self.critic2 = {
                            'sl': q_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, n_tasks['sl'], lr=self.lr['sl']['q']).to(device),
                            'ql': DQN(s_dim-self.dim_excluded['init'], self.n_skills+1, n_tasks['ql'], lr=self.lr['ql']['q']).to(device)
                        }
        self.v = {
                            'sl': v_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), n_tasks['sl'], lr=self.lr['sl']['v']).to(device),
                            'ql': v_Net(s_dim-self.dim_excluded['init'], n_tasks['ql'], lr=self.lr['ql']['v']).to(device)
                        }
        self.v_target = {
                            'sl': v_Net(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), n_tasks['sl'], lr=self.lr['sl']['v']).to(device),
                            'ql': v_Net(s_dim-self.dim_excluded['init'], n_tasks['ql'], lr=self.lr['ql']['v']).to(device)
                        }
        self.actor = s_Net(self.n_skills, s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, lr=self.lr['sl']['pi']).to(device)

        self.memory = {
                        'sl':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'ql':  Memory(self.params['memory_capacity'], n_seed=self.seed)
                    }

        updateNet(self.v_target['sl'], self.v['sl'],1.0)
        updateNet(self.critic2['ql'], self.critic1['ql'],1.0)    
    
    def memorize(self, event, learning_type, init=False):
        if init:
            self.memory[learning_type].store(event[np.newaxis,:])
        else:
            self.memory[learning_type].store(event.tolist())
    
    def decide(self, state, task, explore=True):
        skill = self.decide_q_dist(state, task, explore=explore) if self.decision_type == 'q_dist' else self.decide_epsilon(state, task, explore=explore) 
        return skill

    def decide_q_dist(self, state, task, explore=True):
        s_cuda = torch.FloatTensor(state[self.dim_excluded['init']:]).to(device).view(1,-1)
        q = self.critic1['ql'](s_cuda).squeeze(0)[task,:] if np.random.rand() > 0.5 else self.critic2['ql'](s_cuda).squeeze(0)[task,:]
        with torch.no_grad():
            pi = torch.exp((q-q.max())/(self.alpha['ql'][task]+1e-6)).view(-1)
            pi = pi / pi.sum()
            skill = Categorical(probs=pi).sample().item() if explore else pi.argmax().item()
            return skill

    def decide_epsilon(self, state, task, explore=True):
        s_cuda = torch.FloatTensor(state[self.dim_excluded['init']:]).to(device).view(1,-1)
        with torch.no_grad():
            q = self.critic1['ql'](s_cuda).squeeze(0)[task,:]
            epsilon = self.epsilon if explore else 0.0
            skill = q.argmax().item() if np.random.rand() > epsilon else np.random.randint(self.n_skills+1)
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
        batch = self.memory['ql'].sample(self.batch_size['ql'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['init']:self.s_dim]).to(device)
            A_batch = batch[:,self.s_dim].astype('int')
            r_batch = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.s_dim+2+self.dim_excluded['init']:2*self.s_dim+2]).to(device)
            d_batch = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T_batch = batch[:,2*self.s_dim+3].astype('int')  

            # Optimize q networks
            q = self.critic1['ql'](s_batch)[np.arange(batch_size), T_batch, A_batch]
            qn = self.critic2['ql'](ns_batch)[np.arange(batch_size), T_batch, :]
            # q2 = self.critic2['ql'](s_batch)
            # q = torch.min(torch.stack([q1, q2]), 0)[0].detach()
                           
            q_approx = r_batch + self.gamma * qn.max(1, keepdim=True)[0] * (1.0-d_batch)
            
            q_loss = (((q - q_approx.detach()).clamp(-1.0,1.0))**2).mean()
            self.critic1['ql'].optimizer.zero_grad()
            q_loss.backward()
            clip_grad_norm_(self.critic1['ql'].parameters(), self.clip_value)
            self.critic1['ql'].optimizer.step()

            updateNet(self.critic2['ql'], self.critic1['ql'], self.lr['ql']['v_target'])

            # q2_loss = self.critic2['ql'].loss_func(q2_AT, q_approx.detach())
            # self.critic2['ql'].optimizer.zero_grad()
            # q2_loss.backward()
            # clip_grad_norm_(self.critic2['ql'].parameters(), self.clip_value)
            # self.critic2['ql'].optimizer.step()

            # Anneal epsilon
            self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])
    
    def learn_DQN_SAC(self, only_metrics=False):
        batch = self.memory['ql'].sample(self.batch_size['ql'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['init']:self.s_dim]).to(device)
            A_batch = batch[:,self.s_dim].astype('int')
            r_batch = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.s_dim+2+self.dim_excluded['init']:2*self.s_dim+2]).to(device)
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
                q_approx = r_batch + self.gamma * next_v * (1-d_batch)
                
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

            if not only_metrics:
                # Optimize q networks
                q1 = self.critic1['sl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q2 = self.critic2['sl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                next_v = self.v_target['sl'](ns_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q_approx = r_batch + self.gamma * next_v * (1-d_batch)
                
                q1_loss = self.critic1['sl'].loss_func(q1, q_approx.detach())
                self.critic1['sl'].optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1['sl'].parameters(), self.clip_value)
                self.critic1['sl'].optimizer.step()
                
                q2_loss = self.critic2['sl'].loss_func(q2, q_approx.detach())
                self.critic2['sl'].optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2['sl'].parameters(), self.clip_value)
                self.critic2['sl'].optimizer.step()                

            # Optimize v network
            a_batch_A, log_pa_sApT_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch)
            a_batch = a_batch_A[np.arange(batch_size), T_batch, :]
            log_pa_sT = log_pa_sApT_A[np.arange(batch_size), :, T_batch]
            
            q1_off = self.critic1['sl'](s_batch.detach(), a_batch)
            q2_off = self.critic2['sl'](s_batch.detach(), a_batch)
            q_off = torch.min(torch.stack([q1_off, q2_off]), 0)[0]
            
            v_approx = (q_off - self.alpha['sl'].view(1,-1) * log_pa_sT)[np.arange(batch_size), T_batch].view(-1,1) 

            if not only_metrics:
                v = self.v['sl'](s_batch)[np.arange(batch_size), T_batch].view(-1,1)
            
            task_mask = torch.zeros(batch_size, self.n_tasks['sl']).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            Ha_sT = -(log_pa_sT * task_mask_distribution).sum(0)
            alpha_gradient = Ha_sT.detach() - self.threshold_entropy_alpha['sl']

            if not only_metrics:
                v_loss = ((v - v_approx.detach())**2).mean()
                self.v['sl'].optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v['sl'].parameters(), self.clip_value)
                self.v['sl'].optimizer.step()
                updateNet(self.v_target['sl'], self.v['sl'], self.lr['sl']['v_target'])

                # Optimize skill network
                pi_loss = (-v_approx).mean()
                self.actor.optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()

                # Optimize dual variable                
                log_alpha = torch.log(self.alpha['sl'] + 1e-6)
                log_alpha -= self.lr['sl']['alpha'] * alpha_gradient
                self.alpha['sl'] = torch.exp(log_alpha).clamp(1e-10, 1e+3)

                self.threshold_entropy_alpha['sl'] = np.max([self.threshold_entropy_alpha['sl'] - self.delta_threshold_entropy_alpha, self.min_threshold_entropy_alpha['sl']])
                    
        else:
            log_pa_sT = torch.zeros(1).to(device)  
            Ha_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(a|s,T)': Ha_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics
    
    def estimate_metrics(self, learning_type):
        with torch.no_grad():
            if learning_type == 'sl':
                metrics = self.learn_skills(only_metrics=True)
            elif learning_type == 'ql':
                metrics = self.learn_DQN(only_metrics=True)
        return metrics
    
    def save(self, common_path, specific_path, learning_type):
        self.params['alpha'] = self.alpha
        self.params['threshold_entropy_alpha'] = self.threshold_entropy_alpha
        self.params['init_epsilon'] = self.epsilon
        
        pickle.dump(self.params,open(common_path+'/agent_params.p','wb'))

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
        if learning_type == 'sl':
            torch.save(self.actor.state_dict(), specific_path+'_actor.pt')
    
    def load(self, common_path, specific_path, learning_type, load_memory=True):
        if load_memory: 
            data_batches = pickle.load(open(common_path+'/data_batches_'+learning_type+'.p','rb'))
            pointer = 0
            for i in range(0, data_batches['l']):
                data = pickle.load(open(common_path+'/memory_'+learning_type+str(i+1)+'.p','rb'))
                self.memory[learning_type].data += data
                pointer += len(data)
            self.memory[learning_type].pointer = pointer % self.memory[learning_type].capacity

        self.actor.load_state_dict(torch.load(specific_path+'_actor.pt'))
        self.actor.eval()

        self.critic1[learning_type].load_state_dict(torch.load(specific_path+'_critic1_'+learning_type+'.pt'))
        self.critic2[learning_type].load_state_dict(torch.load(specific_path+'_critic2_'+learning_type+'.pt'))
        self.v[learning_type].load_state_dict(torch.load(specific_path+'_v_'+learning_type+'.pt'))
        self.v_target[learning_type].load_state_dict(torch.load(specific_path+'_v_target_'+learning_type+'.pt'))

        self.critic1[learning_type].eval()
        self.critic2[learning_type].eval()
        self.v[learning_type].eval()
        self.v_target[learning_type].eval()

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
                            'env_names_sl': ['Hopper-v2'],
                            'env_names_ql': ['Hopper-v2'],
                            'env_names_tl': ['Hopper-v2'],
                            'env_steps_sl': 1,
                            'env_steps_ql': 5, 
                            'grad_steps': 1, 
                            'init_steps': 10000,
                            'max_episode_steps': 1000,
                            'tr_steps_sl': 1000,
                            'tr_steps_ql': 200,
                            'tr_epsd_sl': 1000,
                            'tr_epsd_ql': 10000,
                            'eval_epsd_sl': 2,
                            'eval_epsd_interval': 10,
                            'eval_epsd_ql': 2,
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
                            'tl': self.params['env_names_tl']
                        }
        self.n_tasks = {
                            'sl': len(self.env_names['sl']),
                            'ql': len(self.env_names['ql']),
                            'tl': len(self.env_names['tl'])
                        }
        self.steps = {
                        'env': {
                                'sl': self.params['env_steps_sl'],
                                'ql': self.params['env_steps_ql']
                            },
                        'grad': self.params['grad_steps'],

                        'init': self.params['init_steps'],
                        'tr': {
                                'sl': self.params['tr_steps_sl'],
                                'ql': self.params['tr_steps_ql']
                            }
                    }
        self.epsds = {
            'tr': {
                'sl': self.params['tr_epsd_sl'],
                'ql': self.params['tr_epsd_ql']
            },
            'eval': {
                'sl': self.params['eval_epsd_sl'],
                'ql': self.params['eval_epsd_ql'],
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
        self.t_dim = self.sarsd_dim + 1
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
        next_state, reward, done = self.envs[self.learning_type][self.task].step(action)[:3]  
        done = done and self.reset_when_done
        
        event[:self.s_dim] = state
        event[self.s_dim:self.sa_dim] = action
        event[self.sa_dim] = reward
        event[self.sa_dim+1:self.sars_dim] = next_state
        event[self.sars_dim] = float(done)
        event[self.sarsd_dim] = self.task

        self.agent.memorize(event, self.learning_type)   
        return done

    def interaction(self, remember=True, explore=True, learn=True):  
        event = np.empty(self.t_dim)
        initial_state = self.get_obs()
        state = initial_state.copy()
        final_state = initial_state.copy()
        total_reward = 0.0
        done = False

        if self.learning_type == 'sl':
            skill = self.task
        elif self.learning_type == 'ql':
            skill = self.agent.decide(state, self.task, explore=explore)

        for env_step in range(0, self.steps['env'][self.learning_type]):
            action = self.agent.act(state, skill, explore=explore if self.learning_type == 'sl' else False)
            scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
            next_state, reward, done_step, _ = self.envs[self.learning_type][self.task].step(scaled_action)
            done = done_step and self.reset_when_done
            total_reward += reward
            final_state = np.copy(next_state)

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.sars_dim] = next_state
            event[self.sars_dim] = float(done)
            event[self.sarsd_dim] = self.task
        
            if remember and self.learning_type == 'sl': self.agent.memorize(event.copy(), self.learning_type)
            if done: break
            if env_step < self.steps['env'][self.learning_type]-1: state = np.copy(next_state)
        
        if remember and self.learning_type == 'ql':
            event = np.empty(2*self.s_dim+4)
            event[:self.s_dim] = initial_state 
            event[self.s_dim] = skill
            event[self.s_dim+1] = total_reward
            event[self.s_dim+2:2*self.s_dim+2] = final_state
            event[2*self.s_dim+2] = float(done)
            event[2*self.s_dim+3] = self.task

            self.agent.memorize(event.copy(), self.learning_type)

        if learn:
            if self.learning_type == 'sl':
                for _ in range(0, self.steps['grad']):
                    self.agent.learn_skills()
            elif self.learning_type == 'ql':
                for _ in range(0, self.steps['grad']):
                    self.agent.learn_DQN()

        return total_reward, done, event

    def train_agent(self, initialization=True, skill_learning=True, storing_path='', rewards=[], metrics=[], iter_0=0):
        if len(storing_path) == 0: storing_path = self.params['storing_path']

        if initialization:
            self.initialization()
            specific_path = storing_path + '/' + str(0)
            self.save(storing_path, specific_path)
        
        if skill_learning:
            self.train_agent_nets(storing_path=storing_path, rewards=rewards, metrics=metrics, iter_0=iter_0)

        iter_0_ql = iter_0 if not self.learning_type == 'sl' else 0
        self.learning_type = 'ql'
        self.set_envs()
        self.agent.memory['sl'].forget()
        self.train_agent_nets(storing_path=storing_path, iter_0=iter_0_ql)
    
    def train_agent_nets(self, iter_0=0, rewards=[], metrics=[], storing_path=''):        
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
            self.save(storing_path, specific_path)
            np.savetxt(storing_path + '/mean_rewards_'+self.learning_type+'.txt', np.array(rewards))
    
    def train_agent_concepts(self):
        pass

    @property
    def entropy_metric(self):
        return self.learning_type == 'sl' or self.agent.DQN_learning_type == 'SAC'

    def eval_agent_skills(self, eval_epsds=0, explore=False, iter_=0, start_render=False, print_space=True, specific_path='video', max_epsd=0):   
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
            if max_epsd <= 0: max_epsd = (self.envs[self.learning_type][self.task]._max_episode_steps // self.steps['env'][self.learning_type])
            epsd_reward = 0.0

            for eval_step in itertools.count(0):            
                reward, done, event = self.interaction(explore=explore, learn=False)
                event[self.sa_dim] = reward  
                epsd_reward += reward              

                if self.store_video:
                    img = self.envs[self.learning_type][self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif self.render:
                    self.envs[self.learning_type][self.task].render()

                events.append(event)

                if done or (eval_step + 1 >= max_epsd):
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
                stdout.write("Iter %i, epsd %i, epsilon: %f, min r: %i, max r: %i, mean r: %i, epsd r: %i\r " %
                    (iter_, (epsd+1), self.agent.epsilon, min_epsd_reward//1, max_epsd_reward//1, average_reward//1, epsd_reward//1))
            stdout.flush()         

        if print_space: print("")

        if self.store_video: video.release()
        metric_vector = np.array([Ha_sT_average]) if self.entropy_metric else np.array([]) 
        
        self.task = task
        return rewards, np.array(events), metric_vector, np.array(epsd_lenghts)      
    
    def save(self, common_path, specific_path):
        self.params['learning_type'] = self.learning_type
        pickle.dump(self.params, open(common_path+'/params.p','wb'))
        self.agent.save(common_path, specific_path, self.learning_type)
    
    def load(self, common_path, specific_path, load_memory=True):
        self.agent.load(common_path, specific_path, self.learning_type, load_memory=load_memory)

