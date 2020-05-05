import gym
import torch
import numpy as np
import random

from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from gym.utils import seeding

from nets import Memory, v_net, q_net, dueling_q_net, skill_net, classifier_net, RND_module, discrete_AC_mixed, discrete_AC

import os
import time
import pickle
from sys import stdout
import itertools
import curses

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image

width = 1024
height = 768
FPS = 60

fourcc = VideoWriter_fourcc(*'MP42')

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Functions
#############
def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def scale_action(a, min, max):
    return (0.5*(a+1.0)*(max-min) + min)

def set_seed(n_seed):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda": torch.cuda.manual_seed(n_seed)

def is_float(x):
    return isinstance(x, float)

def is_int(x):
    return isinstance(x, int)

def is_tensor(x):
    return isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor)

# Classes
##############
class Agent:
    def __init__(self, s_dim, a_dim, n_tasks, params, seed=0):

        self.params = params.copy()
        default_params = {
            'n_concepts': 10,
            'n_skills': 8,
            'decision_type': 'eps',                            
            'alpha': {
                'sl': 0.1,
                'ql': 0.1,
                'cl': 1e-6,                                        
            },
            'init_epsilon': 1.0,
            'min_epsilon': 0.4,                            
            'delta_epsilon': 2.5e-7,                          
            'init_threshold_entropy_alpha': 0.0,
            'delta_threshold_entropy_alpha': 8e-6,
            'stoA_learning_type': 'SAC',
            'DQL_epsds_target_update': 6000,  
            'joint_sq_learning': False, # TODO: add during agent call
            'target_update_rate': 5e-3,           
            'lr': 3e-4,
            'dims': {
                'init_prop': 2,
                'last_prop': s_dim,
                'init_ext': 3,
                'last_ext': s_dim-60,
            },            
            'batch_size': {
                'sl': 256,
                'ql': 256,
                'tl': 256
            },                            
            'memory_capacity': 1200000,
            'GAE_lambda': 0.95,
            'gamma_E': 0.99,
            'gamma_I': 0.975,                            
            'clip_value': 0.5,
            'factor_I': 1.0,
            'entropy_annealing': False,
            'RND_update_proportion': 0.25,
            'intrinsic_learning': True                                                                          
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

        self.joint = self.params['joint_learning']
        self.n_tasks = n_tasks
        self.seed = seed
        self.n_skills = n_tasks['sl'] if not self.joint else self.params['n_skills']
        self.counter = 0
        self.counter_cl = 0
        v_dim = self.n_skills if not self.joint else n_tasks['ql']  

        self.n_concepts = self.params['n_concepts']
        self.dims = self.params['dims']
        self.batch_size = self.params['batch_size']
        self.lr = self.params['lr']
        self.GAE_lambda = self.params['GAE_lambda']
        self.gamma_E = self.params['gamma_E']
        self.gamma_I = self.params['gamma_I']        
        self.clip_value = self.params['clip_value']
        self.decision_type = self.params['decision_type']
        self.stoA_learning_type = self.params['stoA_learning_type']
        self.DQL_epsds_target_update = self.params['DQL_epsds_target_update']      
        self.entropy_annealing = self.params['entropy_annealing']
        self.RND_update_proportion = self.params['RND_update_proportion']
        self.active_intrinsic_learning = self.params['intrinsic_learning']
        self.target_update_rate = ['target_update_rate']

        self.min_skill_entropy = 0.95 * (-np.log(1/(self.n_skills+1))) # TODO: add to param dict.
        self.novelty_factor = 1.0/np.log(100.0) # TODO: add to param dict.

        # Metric weights
        self.min_threshold_entropy_alpha = -a_dim*1.0/2 # TODO: change name or eliminate if not necessary
        self.threshold_entropy_alpha = self.params['init_threshold_entropy_alpha'] # TODO: change name or eliminate if not necessary
        self.delta_threshold_entropy_alpha = self.params['delta_threshold_entropy_alpha'] # TODO: change name or eliminate if not necessary        
        alpha = self.params['alpha']
        self.alpha = {} # TODO: allow alpha to depend on task
        for learning_type in ['sl', 'ql', 'cl']: # TODO
            self.alpha[learning_type] = alpha[learning_type]
        
        # self.eta = {}
        # eta = self.params['init_eta']
        # for learning_type in ['ql', 'tl']:
        #     self.eta[learning_type] = (eta[learning_type] * torch.ones(self.n_tasks[learning_type]).float().to(device) if is_float(eta[learning_type]) else 
        #                                 (eta[learning_type].float().to(device) if is_tensor(eta[learning_type]) else torch.from_numpy(eta[learning_type]).float().to(device)))
        
        self.epsilon = self.params['init_epsilon']
        self.min_epsilon = self.params['min_epsilon']
        self.delta_epsilon = self.params['delta_epsilon']
        self.factor_I = self.params['factor_I']
               
       # Nets and memory
        self.v = { # TODO: change structure of low level AC
                            'sl': v_net(self.dims['last_ext']-self.dims['init_ext'], v_dim, lr=self.lr).to(device),                            
                        }
        self.v_target = {
                            'sl': v_net(self.dims['last_ext']-self.dims['init_ext'], v_dim, lr=self.lr).to(device),                            
                        }
        if self.stoA_learning_type == 'DQL':
            self.critic1 = {
                                'sl': q_net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr).to(device),
                                'ql': dueling_q_net(self.dims['last_ext']-self.dims['init_ext'], self.n_skills+1, n_tasks['ql'], lr=self.lr).to(device),
                            }
            self.critic2 = {
                                'sl': q_net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr).to(device),
                                'ql': dueling_q_net(self.dims['last_ext']-self.dims['init_ext'], self.n_skills+1, n_tasks['ql'], lr=self.lr).to(device),
                            }
        else:
            self.critic1 = {
                                'sl': q_net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr).to(device),
                                'ql': discrete_AC(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], n_tasks['ql'], lr=self.lr).to(device),
                            }
            self.critic2 = {
                                'sl': q_net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr).to(device),
                            }        

        self.actor = skill_net(self.n_skills, self.dims['last_prop']-self.dims['init_prop'], a_dim, lr=self.lr).to(device)
        self.classifier = classifier_net(self.n_concepts, self.dims['last_ext']-self.dims['init_ext'], self.n_skills+1, n_tasks=self.n_tasks['ql'], lr=3.0e-4).to(device)
        
        self.memory = {
                        'sl':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'ql':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'tl':  Memory(self.params['memory_capacity'], n_seed=self.seed)
                    }
                
        self.NnSdoAST_cl = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1, self.n_concepts).to(device)
        self.NAST_cl = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1).to(device)
        
        self.RND = RND_module(self.dims['last_ext']-self.dims['init_ext'], self.n_tasks['tl'], gamma_I=self.gamma_I).to(device) # TODO: fix when n_tasks['tl'] is 0
        
        self.PA_ST_tl = torch.ones(self.n_concepts, self.n_skills+1).to(device)
        self.NAST_MC = torch.zeros(self.n_concepts, self.n_skills+1).to(device)
        self.QAST_MC = torch.zeros(self.n_concepts, self.n_skills+1).to(device)

        self.transfer_actor = discrete_AC_mixed(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], n_tasks['tl'], self.n_concepts, lr=self.lr).to(device)
       
        updateNet(self.v_target['sl'], self.v['sl'],1.0)
        if self.stoA_learning_type == 'DQL':
            updateNet(self.critic2['ql'], self.critic1['ql'],1.0)    

    def memorize(self, event, learning_type, init=False):
        if init:
            self.memory[learning_type].store(event[np.newaxis,:])
        else:
            self.memory[learning_type].store(event.tolist())
    
    def relate_concept(self, state, explore=True):
        state_cuda = torch.FloatTensor(state[self.dims['init_ext']:self.dims['last_ext']]).to(device).view(1,-1)
        with torch.no_grad():
            return self.classifier.sample_concept(state_cuda, explore=explore)

    def decide(self, state, task, learning_type, explore=True, guess=False, rng=None):
        with torch.no_grad():
            if learning_type == 'ql':                
                if self.stoA_learning_type == 'DQL':
                    skill = self.decide_q_dist(state, task, explore=explore) if self.decision_type == 'q_dist' else self.decide_epsilon(state, task, explore=explore)
                else:
                    s_cuda = torch.FloatTensor(state[self.dims['init_ext']:self.dims['last_ext']]).to(device).view(1,-1)
                    skill = self.critic1['ql'].sample_skill(s_cuda, task, explore=explore, rng=rng)                
                return skill 
            elif learning_type == 'tl':
                s_cuda = torch.FloatTensor(state[self.dims['init_ext']:self.dims['last_ext']]).to(device).view(1,-1)
                skill = self.transfer_actor.sample_skill(s_cuda, task, explore=explore, rng=rng)
                return skill 

    def decide_q_dist(self, state, task, explore=True):
        s_cuda = torch.FloatTensor(state[self.dims['init_ext']:]).to(device).view(1,-1)
        q = self.critic1['ql'](s_cuda).squeeze(0)[task,:] # if np.random.rand() > 0.5 else self.critic2['ql'](s_cuda).squeeze(0)[task,:]
        with torch.no_grad():
            pi = torch.exp((q-q.max())/(self.alpha['ql']+1e-6)).view(-1)
            pi = pi / pi.sum()
            if explore:
                skill = Categorical(probs=pi).sample().item() 
            else:
                tie_breaking_dist = torch.isclose(q, q.max()).float()
                tie_breaking_dist /= tie_breaking_dist.sum()
                skill = Categorical(probs=tie_breaking_dist).sample().cpu() 
            return skill

    def decide_epsilon(self, state, task, explore=True):
        s_cuda = torch.FloatTensor(state[self.dims['init_ext']:]).to(device).view(1,-1)
        with torch.no_grad():
            qe = self.critic1['ql'](s_cuda)
            qe = qe.squeeze(0)[task,:]
            tie_breaking_dist = torch.isclose(qe, qe.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            skill = Categorical(probs=tie_breaking_dist).sample().cpu() 
            skill = skill if np.random.rand() > self.epsilon else np.random.randint(self.n_skills+1)
            return skill            

    def act(self, state, skill, explore=True):
        s_cuda = torch.FloatTensor(state[self.dims['init_prop']:self.dims['last_prop']]).to(device)
        with torch.no_grad():
            a = self.actor.sample_action(s_cuda, skill, explore=explore)if skill < self.n_skills else np.zeros(self.a_dim)
            return a     

    def learn_DQN(self, only_metrics=False):
        if not only_metrics:
            self.learn_DQN_DQL() if self.stoA_learning_type == 'DQL' else self.learn_DQN_SAC('ql', only_metrics=only_metrics)
        else:
            metrics = {} if self.stoA_learning_type == 'DQL' else self.learn_DQN_SAC('ql', only_metrics=only_metrics)
            return metrics

    def learn_DQN_DQL(self):
        self.counter += 1
        batch = self.memory['ql'].sample(self.batch_size['ql'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            A = batch[:,self.s_dim].astype('int')
            re = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
            ns = torch.FloatTensor(batch[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            d = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T = batch[:,2*self.s_dim+3].astype('int')

            # Optimize q networks            
            qe = self.critic1['ql'](s)[np.arange(batch_size), T, A].view(-1,1)
            nqe = self.critic1['ql'](ns)[np.arange(batch_size), T, :]
            nqe_target =  self.critic2['ql'](ns)[np.arange(batch_size), T, :]
            
            best_skills = nqe.argmax(1)
            qe_approx = re/10.0 + self.gamma_E * nqe_target[np.arange(batch_size), best_skills].view(-1,1) * (1.0-d) # +  0.5*RND_error.detach()
            
            q_loss = self.critic1['ql'].loss_func(qe, qe_approx.detach())# + self.critic1['ql'].loss_func(qi_exp, qi_exp_approx.detach()))*IS_weights
            self.critic1['ql'].optimizer.zero_grad()
            q_loss.mean().backward()
            clip_grad_norm_(self.critic1['ql'].parameters(), self.clip_value)
            self.critic1['ql'].optimizer.step()

            if self.counter % self.DQL_epsds_target_update == 0:
                updateNet(self.critic2['ql'], self.critic1['ql'], 1.0)
                self.counter = 0

            # Anneal epsilon
            self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])            
                
    def learn_DQN_SAC(self, learning_type, only_metrics=False, learn_alpha=True):
        batch = self.memory[learning_type].sample(self.batch_size[learning_type])
        batch = np.array(batch)
        batch_size = batch.shape[0]                    
        
        if batch_size > 0:            
            s = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            A = batch[:,self.s_dim].astype('int')
            re = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device) / 10.0
            ns = torch.FloatTensor(batch[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            d = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T = batch[:,2*self.s_dim+3].astype('int')            
            
            # Optimize Q
            qe1, _, qe2, _, PA_sT, log_PA_sT, alpha, log_alpha = self.critic1['ql'](s, T)
            _, nqe1t, _, nqe2t, PnA_nsT, log_PnA_nsT, _, _ = self.critic1['ql'](ns, T)
            
            nqet = torch.min(nqe1t, nqe2t)
            nve = (PnA_nsT * (nqet - alpha.view(-1,1) * log_PnA_nsT)).sum(1, keepdim=True).detach()
            qe_approx = re + self.gamma_E * nve * (1-d)

            qe1_A, qe2_A = qe1[np.arange(batch_size), A].view(-1,1), qe2[np.arange(batch_size), A].view(-1,1)
            qe1_loss = (qe1_A - qe_approx.detach())**2
            qe2_loss = (qe2_A - qe_approx.detach())**2
            
            PA_T = PA_sT.mean(0, keepdim=True)
            HA_sT = -(PA_sT * log_PA_sT).sum(1, keepdim=True)
            HA_sT_mean = HA_sT.detach().mean()
            qt = torch.min(qe1, qe2).detach()
            z = torch.logsumexp(qt.detach()/(alpha+1e-10), 1, keepdim=True)
            
            pi_loss = (PA_sT * (log_PA_sT - (qt/(alpha+1e-10) - z)).detach()).sum(1, keepdim=True)
            if learn_alpha:
                log_pi_target = qt.detach()/(alpha+1e-10) - z
                pi_target = torch.exp(log_pi_target)                  
                H_pi_target = -(pi_target * log_pi_target).sum(1, keepdim=True)
                H_pi_target_mean = H_pi_target.mean()
                scaled_min_entropy = self.min_skill_entropy * self.epsilon
                alpha_loss = log_alpha * (H_pi_target - scaled_min_entropy).detach()

            if not only_metrics:
                self.critic1['ql'].qe1.optimizer.zero_grad()
                qe1_loss.mean().backward()
                clip_grad_norm_(self.critic1['ql'].qe1.parameters(), self.clip_value)
                self.critic1['ql'].qe1.optimizer.step()

                self.critic1['ql'].qe2.optimizer.zero_grad()
                qe2_loss.mean().backward()
                clip_grad_norm_(self.critic1['ql'].qe2.parameters(), self.clip_value)
                self.critic1['ql'].qe2.optimizer.step()
                
                if learn_alpha:
                    self.critic1['ql'].alpha_optim.zero_grad()
                    alpha_loss.mean().backward()
                    self.critic1['ql'].alpha_optim.step()
                    self.critic1['ql'].alpha = self.critic1['ql'].log_alpha.exp()

                self.critic1['ql'].actor.optimizer.zero_grad()
                pi_loss.mean().backward()
                clip_grad_norm_(self.critic1['ql'].actor.parameters(), self.clip_value)
                self.critic1['ql'].actor.optimizer.step()

                self.critic1['ql'].update_targets(self.target_update_rate)  

                # Anneal epsilon
                self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])
                    
        else:
            HA_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(A|s,T)': HA_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics

    def learn_transfer_policy(self, learning_type, only_metrics=False):
        self.CG_SAC_learning(only_metrics=only_metrics)
        
    def CG_SAC_learning(self, only_metrics=False, CG=True, learn_alpha=True):
        batch = self.memory['tl'].sample(self.batch_size['tl'])
        batch = np.array(batch)
        batch_size = batch.shape[0]            
        
        if batch_size > 0:            
            s = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            A = batch[:,self.s_dim].astype('int')
            re = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
            ns = torch.FloatTensor(batch[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            d = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T = batch[:,2*self.s_dim+3].astype('int')  

            T_one_hot = torch.zeros(batch_size, self.n_tasks['tl']).float().to(device)
            T_one_hot[np.arange(batch_size), T] = torch.ones(batch_size).float().to(device)
            
            # Optimize Q
            qe1, _, qe2, _, qi1_exp, _, qi2_exp, _, PA_sT, log_PA_sT, alpha, log_alpha, Alpha, log_Alpha = self.transfer_actor(s, T) # TODO: change name of qi_exp to qi
            _, nqe1t, _, nqe2t, _, _, _, _, PnA_nsT, log_PnA_nsT, _, _, _, _ = self.transfer_actor(ns, T)
            
            nqet = torch.min(nqe1t, nqe2t)
            nve = (PnA_nsT * (nqet - 0.01 * log_PnA_nsT)).sum(1, keepdim=True).detach()
            qe_approx = re + self.gamma_E * nve * (1-d)
            
            qe1_A, qe2_A = qe1[np.arange(batch_size), A].view(-1,1), qe2[np.arange(batch_size), A].view(-1,1)
            qe1_loss = (qe1_A - qe_approx.detach())**2
            qe2_loss = (qe2_A - qe_approx.detach())**2
             
            PA_T = PA_sT.mean(0, keepdim=True)
            HA_sT = -(PA_sT * log_PA_sT).sum(1, keepdim=True)
            HA_sT_mean = HA_sT.detach().mean()
            
            qt = torch.min(qe1, qe2).detach() + self.factor_I * torch.min(qi1_exp, qi2_exp).detach()
            z = torch.logsumexp(qt.detach()/(alpha+1e-10), 1, keepdim=True)

            if CG:
                PS_s = self.classifier(s)[0].detach()                
                ideal_PA_sT = torch.exp(qt/(alpha.detach() + 1e-10) - z).detach()
                ideal_PA_sT = ideal_PA_sT / ideal_PA_sT.sum(1, keepdim=True)
                Z = torch.logsumexp(self.QAST_MC.detach() / (Alpha.view(-1,1) + 1e-10), 1, keepdim=True)
                PA_ST = torch.exp(self.QAST_MC.detach() / (Alpha.view(-1,1) + 1e-10) - Z)
                PA_ST = (PA_ST + 1e-10) / ((PA_ST + 1e-10).sum(1, keepdim=True))                
                      
                log_PA_ST = torch.log(PA_ST + 1e-10) 
                HA_ST = -(PA_ST * log_PA_ST).sum(1)
                HA_ST_mean = (PS_s * HA_ST.view(1,-1)).sum(1, keepdim=True)                

            pi_loss = (PA_sT * (log_PA_sT - (qt/(alpha+1e-10) - z)).detach()).sum(1, keepdim=True)
            if CG or learn_alpha:
                log_novelty_ratios = self.RND.novelty_ratios(s, T[0]).detach()             

            if CG:
                S = PS_s.argmax(1).cpu().numpy()
                HS_s = -(PS_s * torch.log(PS_s + 1e-10)).sum(1, keepdim=True)
                concept_entropy_bottleneck = 1 - HS_s.detach() / np.log(self.n_concepts)
                divergence_per_concept = (PA_sT * (log_PA_sT - log_PA_ST[S,:]).detach()).sum(1, keepdim=True)
                novelty_factor_0 = 1.0 - 1.0 / (1.0 + torch.exp(-2.0 * (log_novelty_ratios - np.log(10)))).view(-1,1)
                total_bottleneck = novelty_factor_0.detach() * concept_entropy_bottleneck
                pi_loss = (1-total_bottleneck) * pi_loss + total_bottleneck * divergence_per_concept            

            if learn_alpha:
                log_pi_target = qt.detach()/(alpha+1e-10) - z
                pi_target = torch.exp(log_pi_target)                  
                H_pi_target = -(pi_target * log_pi_target).sum(1, keepdim=True)
                H_pi_target_mean = H_pi_target.mean()
                scaled_min_entropy = self.min_skill_entropy * (1.0 / (1.0 + self.novelty_factor * log_novelty_ratios))
                alpha_loss = log_alpha * (H_pi_target - scaled_min_entropy.view(-1,1)).detach()

                scaled_min_entropy_mean = scaled_min_entropy.mean()
                active_concepts = (self.QAST_MC.sum(-1) > 0.0).float().view(-1).detach()
                Alpha_loss = log_Alpha.view(-1) * (HA_ST.view(-1) - 0.5*(scaled_min_entropy_mean + np.log(self.n_skills + 1))).detach() * active_concepts
            
            self.transfer_actor.qe1.optimizer.zero_grad()
            qe1_loss.mean().backward()
            clip_grad_norm_(self.transfer_actor.qe1.parameters(), self.clip_value)
            self.transfer_actor.qe1.optimizer.step()

            self.transfer_actor.qe2.optimizer.zero_grad()
            qe2_loss.mean().backward()
            clip_grad_norm_(self.transfer_actor.qe2.parameters(), self.clip_value)
            self.transfer_actor.qe2.optimizer.step()

            if learn_alpha:
                self.transfer_actor.alpha_optim.zero_grad()
                alpha_loss.mean().backward()
                self.transfer_actor.alpha_optim.step()
                self.transfer_actor.alpha = self.transfer_actor.log_alpha.exp()

                self.transfer_actor.Alpha_optim.zero_grad()
                Alpha_loss.mean().backward()
                self.transfer_actor.Alpha_optim.step()
                self.transfer_actor.Alpha = self.transfer_actor.log_Alpha.exp()

            self.transfer_actor.actor.optimizer.zero_grad()
            pi_loss.mean().backward()
            clip_grad_norm_(self.transfer_actor.actor.parameters(), self.clip_value)
            self.transfer_actor.actor.optimizer.step()

            if CG:
                self.PA_ST_tl = PA_ST.detach().clone()    

            self.transfer_actor.update_targets(self.target_update_rate)      

    def MC_learning(self, episode):
        N = len(episode)
        if N > 0:
            G = 0
            returns = torch.zeros((self.n_concepts, self.n_skills + 1)).to(device)
            visited = torch.zeros((self.n_concepts, self.n_skills + 1)).to(device)
            for i in range(N-1, -1, -1):
                S, A, R = episode[i]
                G = self.gamma_E * G + R
                returns[int(S),int(A)] += G
                visited[int(S),int(A)] += 1
            self.NAST_MC = (1-0.1) * self.NAST_MC + visited
            self.QAST_MC = (self.QAST_MC + (returns - visited * self.QAST_MC)/self.NAST_MC.clamp(1.0,np.infty)).detach().clone()                     

    def intrinsic_learning(self, trajectory, reset=False):
        N = len(trajectory) 
        if N > 0:
            trajectory = np.array(trajectory)
            s = torch.FloatTensor(trajectory[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            A = trajectory[:,self.s_dim].astype('int')
            ns = torch.FloatTensor(trajectory[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            # d = torch.FloatTensor(trajectory[:,2*self.s_dim+2]).to(device).view(-1,1)
            T = int(trajectory[0,2*self.s_dim+3])

            self.RND.update_obs_rms(s, T)
            ri_exp = self.RND(ns, T).sum(1, keepdim=True)
            self.r_max = max(self.r_max, ri_exp.max().item())
            
            rffs_int = torch.FloatTensor([self.RND.rff_int.update(rew) for rew in ri_exp.detach().squeeze().tolist()]).to(device)
            self.RND.rff_rms_int.update(rffs_int, T)
            ri_exp_normalized = ri_exp.detach() / self.RND.rff_rms_int.var.sqrt()

            mask = torch.rand(len(ri_exp)).to(device)
            mask = (mask < self.RND_update_proportion).type(torch.FloatTensor).to(device)
            intrinsic_loss = (ri_exp * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))

            self.RND.predictor.optimizer.zero_grad()
            intrinsic_loss.mean().backward()
            clip_grad_norm_(self.RND.predictor.parameters(), self.clip_value)
            self.RND.predictor.optimizer.step()

            with torch.no_grad():
                pi_end = self.transfer_actor.actor(ns[-1,:].view(1,-1))[0].squeeze(0)[T, :]
                pi_old, log_pi_old = self.transfer_actor.actor(s)
                pi_old, log_pi_old = pi_old.detach()[:, T, :], log_pi_old.detach()[:, T, :]
                PA_T = pi_old.mean(0, keepdim=True)
                HA_sT = -(pi_old * log_pi_old).sum(1, keepdim=True)
                HA_sT_mean = HA_sT.mean()            
                        
                qi1_end = self.transfer_actor.qi1_exploration(ns[-1,:].view(1,-1)).squeeze(0)[T, :]
                qi2_end = self.transfer_actor.qi2_exploration(ns[-1,:].view(1,-1)).squeeze(0)[T, :]
                qi_end = torch.min(qi1_end, qi2_end)
                vi_end = (pi_end * qi_end).sum()

                qi1_exp = self.transfer_actor.qi1_exploration(s)[:,T,:]
                qi2_exp = self.transfer_actor.qi2_exploration(s)[:,T,:]
                qi_exp = torch.min(qi1_exp, qi2_exp)
                vi_exp = (pi_old * qi_exp).sum(1, keepdim=True)
                
            return_i = torch.zeros_like(ri_exp)
            lastGAE = 0.0
            for t in range(N-1, -1, -1):
                next_val = vi_exp[t+1,:] if t+1<N else vi_end
                delta = ri_exp_normalized[t,:] + self.gamma_I * next_val - vi_exp[t,:]
                lastGAE = delta + (self.gamma_I * self.GAE_lambda) * lastGAE
                return_i[t,:] = lastGAE + vi_exp[t,:]

            for _ in range(0, 4):
                qi1_exp_A = self.transfer_actor.qi1_exploration(s)[np.arange(N),T,A].view(-1,1)
                qi2_exp_A = self.transfer_actor.qi2_exploration(s)[np.arange(N),T,A].view(-1,1)
                qi1_exp_loss = (qi1_exp_A - return_i.detach())**2
                qi2_exp_loss = (qi2_exp_A - return_i.detach())**2

                self.transfer_actor.qi1_exploration.optimizer.zero_grad()
                qi1_exp_loss.mean().backward()
                clip_grad_norm_(self.transfer_actor.qi1_exploration.parameters(), self.clip_value)
                self.transfer_actor.qi1_exploration.optimizer.step()

                self.transfer_actor.qi2_exploration.optimizer.zero_grad()
                qi2_exp_loss.mean().backward()
                clip_grad_norm_(self.transfer_actor.qi2_exploration.parameters(), self.clip_value)
                self.transfer_actor.qi2_exploration.optimizer.step()
    
    def learn_skills(self, only_metrics=False):
        batch = self.memory['sl'].sample(self.batch_size['sl'])
        batch = np.array(batch)
        batch_size = batch.shape[0]        

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            s_batch_prop = torch.FloatTensor(batch[:,self.dims['init_prop']:self.dims['last_prop']]).to(device)
            a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
            r_batch = torch.FloatTensor(batch[:,self.sa_dim]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.sa_dim+1+self.dims['init_ext']:self.sa_dim+1+self.dims['last_ext']]).to(device)
            d_batch = torch.FloatTensor(batch[:,self.sars_dim]).view(-1,1).to(device)
            T_batch = batch[:,self.sarsd_dim].astype('int')  

            if not only_metrics:
                # Optimize q networks
                q1_E = self.critic1['sl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q2_E = self.critic2['sl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                next_v_E = self.v_target['sl'](ns_batch)[np.arange(batch_size), T_batch].view(-1,1)                

                q_approx_E = r_batch + self.gamma_E * next_v_E * (1-d_batch)
                
                q1_loss = self.critic1['sl'].loss_func(q1_E, q_approx_E.detach())
                self.critic1['sl'].optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1['sl'].parameters(), self.clip_value)
                self.critic1['sl'].optimizer.step()
                
                q2_loss = self.critic2['sl'].loss_func(q2_E, q_approx_E.detach())
                self.critic2['sl'].optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2['sl'].parameters(), self.clip_value)
                self.critic2['sl'].optimizer.step()       

            # Optimize v network
            a_batch_A, log_pa_sApT_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch_prop.detach())
            if not self.joint:
                A_batch = T_batch
            else:
                A_batch = self.critic1['ql'].sample_skills(s_batch, T_batch)
            a_batch_off = a_batch_A[np.arange(batch_size), A_batch, :]
            log_pa_sT = log_pa_sApT_A[np.arange(batch_size), A_batch].view(-1,1)
            
            q1_off_E = self.critic1['sl'](s_batch.detach(), a_batch_off)
            q2_off_E = self.critic2['sl'](s_batch.detach(), a_batch_off)
            q_off_E = torch.min(torch.stack([q1_off_E, q2_off_E]), 0)[0][np.arange(batch_size), T_batch].view(-1,1)
            
            v_approx_E = q_off_E - self.alpha['sl'] * log_pa_sT

            if not only_metrics:
                v_E = self.v['sl'](s_batch)[np.arange(batch_size), T_batch].view(-1,1)
            
            task_mask = torch.zeros(batch_size, self.n_skills).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            Ha_sT = -(log_pa_sT * task_mask_distribution).sum(0)
            if self.entropy_annealing: alpha_gradient = Ha_sT.detach() - self.threshold_entropy_alpha

            if not only_metrics:
                v_loss = self.v['sl'].loss_func(v_E.view(-1,1), v_approx_E.view(-1,1).detach())
                self.v['sl'].optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v['sl'].parameters(), self.clip_value)
                self.v['sl'].optimizer.step()
                updateNet(self.v_target['sl'], self.v['sl'], self.target_update_rate)
                
                # Optimize skill network
                self.actor.optimizer.zero_grad()                
                pi_loss = -(v_approx_E).mean()                  
                pi_loss.backward()                                 
                
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()                    

                # Optimize dual variable
                if self.entropy_annealing:                                    
                    log_alpha = torch.log(self.alpha['sl'] + 1e-6)
                    log_alpha -= self.lr * alpha_gradient
                    self.alpha['sl'] = torch.exp(log_alpha).clamp(1e-10, 1e+3)

                    self.threshold_entropy_alpha = np.max([self.threshold_entropy_alpha - self.delta_threshold_entropy_alpha, self.min_threshold_entropy_alpha])
                    
        else:
            log_pa_sT = torch.zeros(1).to(device)  
            Ha_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(a|s,T)': Ha_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics    
   
    def learn_concepts(self):
        batch_list = self.memory['ql'].sample(self.batch_size['ql']*2)
        batch = np.array(batch_list)
        del batch_list
        
        batch_size = batch.shape[0]

        if batch_size > 0:
            s = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            A = batch[:,self.s_dim].astype('int')
            ns = torch.FloatTensor(batch[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            T = batch[:,2*self.s_dim+3].astype('int')

            if self.stoA_learning_type == 'DQL':
                q = self.critic1['ql'](s)[np.arange(batch_size), T, :]
                PA_sT = torch.exp((q-q.max(1, keepdim=True)[0])/1.0)
                PA_sT = PA_sT / PA_sT.sum(1, keepdim=True)
            else:
                PA_sT = self.critic1['ql'].actor(s)[0][np.arange(s.shape[0]), T, :]
            A_off = PA_sT.argmax(1)

            A_off_one_hot = torch.zeros(batch_size, self.n_skills+1).to(device)
            A_off_one_hot[np.arange(batch_size), A_off] = torch.ones(batch_size,).to(device)
            
            A_one_hot = torch.zeros(batch_size, self.n_skills+1).to(device)
            A_one_hot[np.arange(batch_size), A] = torch.ones(batch_size,).to(device)

            T_one_hot = torch.zeros(batch_size, self.n_tasks['cl']).to(device)
            T_one_hot[np.arange(batch_size), T] = torch.ones(batch_size,).to(device)
            
            PT = T_one_hot.sum(0) + 1e-10
            PT = PT.view(-1,1) / PT.sum()
            PA_T_data = (A_one_hot.unsqueeze(1) * T_one_hot.unsqueeze(2)).sum(0) + 1e-10
            PA_T_data /= PA_T_data.sum(1, keepdim=True)
            
            PS_s, log_PS_s = self.classifier(s)
            PnS_ns = self.classifier(ns)[0]
            
            NAST_new = ((PS_s.unsqueeze(2) * PA_sT.unsqueeze(1)).unsqueeze(1) * T_one_hot.unsqueeze(2).unsqueeze(3)).sum(0)
            NnSAST_new = (((PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)).unsqueeze(1) * T_one_hot.unsqueeze(2).unsqueeze(3)).unsqueeze(4) * PnS_ns.unsqueeze(1).unsqueeze(2).unsqueeze(3)).sum(0)
            df1 = 0.01 * (self.NAST_cl.sum(2, keepdim=True) >= 1.0).float() # TODO: add to params
            df2 = 0.01 * (self.NnSdoAST_cl.sum(3, keepdim=True) >= 1.0).float()
            NAST = (1-df1) * self.NAST_cl + NAST_new
            NnSAST = (1-df2) * self.NnSdoAST_cl + NnSAST_new

            PS_T = NAST.sum(2) / NAST.sum(2).sum(1, keepdim=True)
            PA_ST = NAST / NAST.sum(2, keepdim=True)
            PnS_TdoA = NnSAST.sum(1) / NnSAST.sum(1).sum(2, keepdim=True)
            PnS_STdoA = NnSAST / NnSAST.sum(3, keepdim=True)
            PA_T = NnSAST.sum((1,3)) / NnSAST.sum((1,3)).sum(1, keepdim=True)
            PA_T_policy = NAST.sum(1) / NAST.sum(1).sum(1, keepdim=True)

            log_PS_T = torch.log(PS_T+1e-10)
            log_PA_ST = torch.log(PA_ST + 1e-10)            
            log_PnS_TdoA = torch.log(PnS_TdoA+1e-10)
            log_PnS_STdoA = torch.log(PnS_STdoA + 1e-10)
            
            HS_T = -(PS_s * log_PS_T[T, :]).sum(1).mean()
            HS_s = -(PS_s * log_PS_s).sum(1).mean()
            ISs_T = HS_T - HS_s
            
            IS_factor = (1/((self.n_skills+1) * PA_T[T, A])).view(-1,1).detach()
                        
            HA_T = -(PA_T_policy * torch.log(PA_T_policy+1e-10)).sum(1).mean()
            HA_ST = -((log_PA_ST[T,:,:] * PA_sT.unsqueeze(1)).sum(2) * PS_s).sum(1).mean()            
            ISA_T = HA_T.detach() - HA_ST
            
            HnS_TdoA_wIS = -(PS_s * (PnS_ns * log_PnS_TdoA.detach()[T, A, :]).sum(1, keepdim=True)).sum(1).mean()
            HnS_TdoA = -(PnS_ns * log_PnS_TdoA[T, A, :]* IS_factor.view(-1,1)).sum(1).mean() #  * IS_factor.view(-1,1)
            HnS_STdoA = -(PS_s * (PnS_ns.unsqueeze(1) * log_PnS_STdoA[T, :, A, :] * IS_factor.view(-1,1,1)).sum(2)).sum(1).mean() #  * IS_factor.view(-1,1,1)
            InSS_TdoA = HnS_TdoA - HnS_STdoA
            
            beta1 = 1.0e-1 # TODO: add to params
            beta2 = 0.25e-1 # TODO: add to params
            alpha2 = 1.0*beta1 / (1-beta2) # TODO: add to params

            classification_loss = -torch.logsumexp(log_PA_ST[T, :, A_off] + log_PS_s, dim=1).mean()
            model_loss = -torch.logsumexp(torch.log((PnS_STdoA[T, :, A, :] * PnS_ns.unsqueeze(1)).sum(2)+1e-10) + log_PS_s, dim=1).mean()

            classifier_loss = (beta1 + alpha2*beta2) * ISs_T - ISA_T - alpha2 * InSS_TdoA 
            classifier_loss_norm = classifier_loss

            self.classifier.optimizer.zero_grad()
            classifier_loss_norm.backward()
            clip_grad_norm_(self.classifier.parameters(), self.clip_value)
            self.classifier.optimizer.step()

            self.NAST_cl = NAST.detach().clone()
            self.NnSdoAST_cl = NnSAST.detach().clone()
            
            return(classifier_loss.detach().item(), 
                    HS_T.detach().item(),
                    HS_s.detach().item(),
                    ISs_T.detach().item(),
                    HA_ST.detach().item(),
                    HA_T.detach().item(),
                    ISA_T.detach().item(),
                    HnS_STdoA.detach().item(),
                    HnS_TdoA.detach().item(),
                    InSS_TdoA.detach().item(),
                    classification_loss.detach().item(),
                    model_loss.detach().item())     
    
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
        self.params['init_threshold_entropy_alpha'] = self.threshold_entropy_alpha
        self.params['init_epsilon'] = self.epsilon
        
        pickle.dump(self.params,open(common_path+'/agent_params.p','wb'))

        if learning_type in ['sl', 'ql']:
            data_batches = {'l': len(self.memory[learning_type].data)//20000+1}
            for i in range(0, data_batches['l']):
                if i+1 < data_batches['l']:
                    pickle.dump(self.memory[learning_type].data[20000*i:20000*(i+1)],open(common_path+'/memory_'+learning_type+str(i+1)+'.p','wb'))
                else:
                    pickle.dump(self.memory[learning_type].data[20000*i:-1],open(common_path+'/memory_'+learning_type+str(i+1)+'.p','wb'))
            pickle.dump(data_batches,open(common_path+'/data_batches_'+learning_type+'.p','wb'))

            if learning_type in ['sl', 'ql']:
                torch.save(self.critic1[learning_type].state_dict(), specific_path+'_critic1_'+learning_type+'.pt')
                if self.stoA_learning_type == 'DQL':
                    torch.save(self.critic2[learning_type].state_dict(), specific_path+'_critic2_'+learning_type+'.pt')
            
            if learning_type == 'sl':
                torch.save(self.v[learning_type].state_dict(), specific_path+'_v_'+learning_type+'.pt')
                torch.save(self.v_target[learning_type].state_dict(), specific_path+'_v_target_'+learning_type+'.pt')
                torch.save(self.actor.state_dict(), specific_path+'_actor_'+learning_type+'.pt')

        if learning_type == 'tl':
            torch.save(self.transfer_actor.state_dict(), specific_path+'_transfer_actor_'+learning_type+'.pt')
            torch.save(self.RND.state_dict(), specific_path+'_RDN_'+learning_type+'.pt')
            pickle.dump(self.NAST_MC,open(specific_path+'_NAST_MC_'+learning_type+'.p','wb'))
            pickle.dump(self.QAST_MC,open(specific_path+'_QAST_MC_'+learning_type+'.p','wb'))
            pickle.dump(self.PA_ST_tl,open(specific_path+'_PA_ST_'+learning_type+'.p','wb'))

        elif learning_type == 'cl':
            torch.save(self.classifier.state_dict(), specific_path+'_classifier.pt')
            pickle.dump(self.NAST_cl,open(specific_path+'_NAST_'+learning_type+'.p','wb'))            
            pickle.dump(self.NnSdoAST_cl,open(specific_path+'_NnSdoAST_'+learning_type+'.p','wb'))
    
    def load(self, common_path, specific_path, learning_type, load_memory=True):
        if learning_type in ['sl', 'ql' ]:
            if load_memory: 
                data_batches = pickle.load(open(common_path+'/data_batches_'+learning_type+'.p','rb'))
                pointer = 0
                for i in range(0, data_batches['l']):
                    try:
                        data = pickle.load(open(common_path+'/memory_'+learning_type+str(i+1)+'.p','rb'))
                        self.memory[learning_type].data += data
                        pointer += len(data)
                    except:
                        pass
                self.memory[learning_type].pointer = pointer % self.memory[learning_type].capacity
                self.memory[learning_type].data = self.memory[learning_type].data[-self.memory[learning_type].capacity:]

            if learning_type == 'sl': 
                self.v[learning_type].load_state_dict(torch.load(specific_path+'_v_'+learning_type+'.pt'))
                self.v_target[learning_type].load_state_dict(torch.load(specific_path+'_v_target_'+learning_type+'.pt'))
                self.v[learning_type].train()
                self.v_target[learning_type].train()  

                self.actor.load_state_dict(torch.load(specific_path+'_actor_'+learning_type+'.pt'))
                self.actor.train()

            if learning_type in ['sl', 'ql']:
                self.critic1[learning_type].load_state_dict(torch.load(specific_path+'_critic1_'+learning_type+'.pt'))
                if self.stoA_learning_type == 'DQL':
                    self.critic2[learning_type].load_state_dict(torch.load(specific_path+'_critic2_'+learning_type+'.pt'))

                self.critic1[learning_type].train()
                if self.stoA_learning_type == 'DQL':
                    self.critic2[learning_type].train()                           

        if learning_type == 'tl':
            self.transfer_actor.load_state_dict(torch.load(specific_path+'_transfer_actor_'+learning_type+'.pt'))
            self.RND.load_state_dict(torch.load(specific_path+'_RDN_'+learning_type+'.pt'))   
            self.RND.train()

        if learning_type == 'cl':
            self.classifier.load_state_dict(torch.load(specific_path+'_classifier.pt'))
            self.classifier.eval()
            try:
                self.PS_T = pickle.load(open(specific_path+'_PS_T_'+learning_type+'.p','rb'))
                self.PA_ST = pickle.load(open(specific_path+'_PA_ST_'+learning_type+'.p','rb'))
            except:
                pass
            self.NAST_cl = pickle.load(open(specific_path+'_NAST_'+learning_type+'.p','rb'))            
            self.NnSdoAST_cl = pickle.load(open(specific_path+'_NnSdoAST_'+learning_type+'.p','rb'))
            
            try:
                self.PnS_STdoA = pickle.load(open(specific_path+'_PnS_STdoA_'+learning_type+'.p','rb'))
            except:
                self.PnS_STdoA = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1, self.n_concepts).to(device) / self.n_concepts

    def classify(self, T=0, path='', data=None):
        if data is None:
            data = self.memory['ql'].data
            task_data = [i for i in data if int(i[2*self.s_dim+3]) == T]
            task_data = np.array(task_data)            
        else:
            task_data = np.array(data)
        data_size = task_data.shape[0]
        
        if data_size > 0:
            s = task_data[:,:self.s_dim]
            s_cuda = torch.FloatTensor(s[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            PS_s = self.classifier(s_cuda)[0]
            numpy_PS_s = PS_s.detach().cpu().numpy()
            S = numpy_PS_s.argmax(1).reshape(-1,1)
                    
            x, y, q = s[:,0].reshape(-1,1), s[:,1].reshape(-1,1), s[:,3:7]
            cos_half_theta = 1-2*(q[:,2]**2+q[:,3]**2)
            sin_half_theta = 2*(q[:,0]*q[:,3] + q[:,1]*q[:,2])
            theta = np.arctan2(sin_half_theta, cos_half_theta).reshape(-1,1)

            angles = np.linspace(-np.pi, np.pi, 9)
            deltas = theta - angles.reshape(1,-1)
            group_id = (np.abs(deltas) <= np.pi/8).argmax(1).reshape(-1,1)
            group_id = np.array([group_id[i] if group_id[i] != 8 else 0 for i in range(0, data_size)])
            export_data = np.concatenate((x, y, theta, S, group_id.reshape(-1,1), numpy_PS_s), axis=1)
            np.savetxt(path + 'classified_samples_'+str(T)+'.txt', export_data)
            print("Samples classified")
    
    def restart_policies(self):        
        self.critic1['ql'] = discrete_AC(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], self.n_tasks['ql'], lr=self.lr).to(device)

class System:
    def __init__(self, params, agent_params={}, skill_learning=True):
        
        self.params = params
        default_params = {
            'seed': 1000,
            'joint_learning': False,
            'joint_cycles': 200,
            'env_names_sl': [],
            'env_names_ql': [],
            'env_names_tl': [],
            'env_steps_sl': 1,
            'env_steps_ql': 5,
            'env_steps_tl': 5, 
            'grad_steps': 1, 
            'init_steps': 10000,
            'max_episode_steps': 1000,
            'tr_steps_sl': 1000,
            'tr_steps_ql': 600,
            'tr_epsd_sl': 4000,
            'tr_epsd_ql': 6000,
            'tr_epsd_wu': 40,
            'tr_epsd_tl': 100,
            'tr_steps_cl': 100000,
            'tr_steps_tl': 100,
            'eval_epsd_sl': 10,
            'eval_epsd_interval': 20,
            'eval_epsd_ql': 5,
            'eval_epsd_tl': 5,
            'eval_steps_sl': 1000,
            'eval_steps_ql': 600,
            'eval_steps_tl': 1800,
            'batch_size': 256, 
            'render': True, 
            'reset_when_done': True, 
            'store_video': False,
            'storing_path': '',
            'MT_steps': 200,
            'update_steps_tl': 4,
            'active_RND': True,
            'masked_done': True,
            'active_MC': True                           
        }

        for key, value in default_params.items():
            if key not in self.params.keys():
                self.params[key] = value

        self.seed = self.params['seed']
        set_seed(self.seed)
        self.np_random, _ = seeding.np_random(self.seed)
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
            },
            'MT': self.params['MT_steps'],
            'update': {
                'tl': self.params['update_steps_tl']
            },
            'eval': {
                'sl': self.params['eval_steps_sl'],
                'ql': self.params['eval_steps_ql'],
                'tl': self.params['eval_steps_tl']
            }
        }
        self.epsds = {
            'tr': {
                'sl': self.params['tr_epsd_sl'],
                'ql': self.params['tr_epsd_ql'],
                'tl': self.params['tr_epsd_tl'],
                'wu': self.params['tr_epsd_wu']
            },
            'eval': {
                'sl': self.params['eval_epsd_sl'],
                'ql': self.params['eval_epsd_ql'],
                'tl': self.params['eval_epsd_tl'],
                'interval': self.params['eval_epsd_interval']
            },
        }
        self.joint_cycles = self.params['joint_cycles']
       
        self.batch_size = self.params['batch_size']
        self.render = self.params['render']
        self.store_video = self.params['store_video']
        self.reset_when_done = self.params['reset_when_done']
        self._max_episode_steps = self.params['max_episode_steps']
        
        self.envs = {}
        self.joint = self.params['joint_learning']
        self.active_RND = self.params['active_RND']
        self.active_MC = self.params['active_MC']
        self.masked_done = self.params['masked_done']
        self.learning_type = 'sl' if not self.joint else 'ql' # if skill_learning else 'ql'

        self.set_envs()

        self.s_dim = self.envs[self.learning_type][0].observation_space.shape[0]
        self.a_dim = self.envs[self.learning_type][0].action_space.shape[0]        
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = self.s_dim*2 + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 1
        self.epsd_counter = 0
        self.task = 0
        self.MT_task = 0

        self.min_action = self.envs[self.learning_type][0].action_space.low[0]
        self.max_action = self.envs[self.learning_type][0].action_space.high[0]

        n_tasks = self.n_tasks.copy()
        self.multitask_envs = {
            'sl': False,
            'ql': False,  # TODO
            'tl': False,
        } 
        self.check_multitask(n_tasks)
        agent_params['joint_learning'] = self.joint
        self.agent = Agent(self.s_dim, self.a_dim, n_tasks, agent_params, seed=self.seed)               

    def check_multitask(self, n_tasks):
        if self.n_tasks[self.learning_type] == 1:
            try:
                n = self.envs[self.learning_type][0]._n_tasks
                n_tasks[self.learning_type] = n
                self.multitask_envs[self.learning_type] = True
                self.n_MT_tasks = n
            except:
                pass 

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
        if change_env: self.task = (self.task+1) % self.n_tasks[self.learning_type]
        self.envs[self.learning_type][self.task].reset()        
    
    def get_obs(self):
        state = self.envs[self.learning_type][self.task]._get_obs().copy()
        return state
     
    def initialization(self, init_steps=0):         
        self.reset()
        skill = 0
        if init_steps == 0: init_steps = self.steps['init']
        for init_step in range(0, init_steps * self.n_tasks[self.learning_type]):
            if self.multitask_envs[self.learning_type]:
                if (init_step % self.steps['env'][self.learning_type]) == 0 and np.random.rand()>0.95:
                    skill = np.random.randint(self.agent.n_skills)
            else:
                skill = self.task
            done = self.interaction_init(skill)
            limit_reached = (init_step+1) % init_steps == 0
            if done or limit_reached: self.reset(change_env=limit_reached)
            if self.render: self.envs[self.learning_type][self.task].render()                        
        print("Finished initialization...")

    def interaction_init(self, skill):  
        event = np.empty(self.t_dim)
        state = self.get_obs()
        action = 2.0*np.random.rand(self.a_dim)-1.0        
        next_state, reward, done, info = self.envs[self.learning_type][self.task].step(action)  
        done = done and self.reset_when_done
        if self.multitask_envs[self.learning_type] and self.learning_type == 'sl':
            skill_reward = info['reward_'+str(skill)]
            reward += skill_reward
        
        if self.multitask_envs[self.learning_type] and self.learning_type == 'ql':
            skill_reward = info['reward_'+str(self.MT_task)]
            reward += skill_reward

        event[:self.s_dim] = state
        event[self.s_dim:self.sa_dim] = action
        event[self.sa_dim] = reward
        event[self.sa_dim+1:self.sars_dim] = next_state
        event[self.sars_dim] = float(done)
        event[self.sarsd_dim] = skill
        
        self.agent.memorize(event, 'sl')   
        return done

    def interaction(self, remember=True, explore=True, learn=True, lim=0, previous_skill = 0, joint_warmup=False):  
        event = np.empty(self.t_dim)
        initial_state = self.get_obs()
        state = initial_state.copy()
        final_state = initial_state.copy()
        total_reward = 0.0
        done = end_step = False
        max_env_step = self.steps['env'][self.learning_type]
        
        task = self.MT_task if self.multitask_envs[self.learning_type] else self.task

        try:
            self.envs[self.learning_type][self.task]._update_quaternion()
        except:
            pass

        if self.learning_type == 'sl':
            if self.multitask_envs[self.learning_type]:
                if np.random.rand() > 0.95:
                    skill = np.random.randint(self.agent.n_skills)
                else:
                    skill = previous_skill
            else:
                skill = task
        elif self.learning_type == 'ql':            
            skill = self.agent.decide(state, task, self.learning_type, explore=explore)
        elif self.learning_type == 'tl':
            if remember:
                skill = self.agent.decide(state, task, self.learning_type, explore=explore) # TODO
            else:
                skill = self.agent.decide(state, task, self.learning_type, explore=explore, rng=self.np_random)   
            s_cuda = torch.FloatTensor(state[self.agent.dims['init_ext']:self.agent.dims['last_ext']]).to(device).view(1,-1)
            with torch.no_grad():
                concept = self.agent.classifier(s_cuda)[0].argmax().item()
            
        if self.env_names[self.learning_type][self.task] == 'AntCrossMaze-v3':
            self.envs[self.learning_type][self.task]._update_led_visualization(concept, skill)
        
        for env_step in itertools.count(0):
            action = self.agent.act(state, skill, explore=explore if (self.learning_type == 'sl' or self.joint and self.learning_type=='ql') else False)
            scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
            next_state, reward, done, info = self.envs[self.learning_type][self.task].step(scaled_action)
            if self.multitask_envs[self.learning_type] and self.learning_type == 'sl':
                skill_reward = info['reward_'+str(skill)]
                reward += skill_reward
            if self.multitask_envs[self.learning_type] and self.learning_type == 'ql':
                try:
                    task_reward = info['reward_'+str(self.MT_task)]
                    reward += task_reward
                except:
                    pass
            end_step = end_step or (done and self.reset_when_done)
            total_reward += reward 
            final_state = np.copy(next_state)

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.sars_dim] = next_state
            event[self.sars_dim] = float(done)
            if not self.joint:  # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
                event[self.sarsd_dim] = skill
            else:
                event[self.sarsd_dim] = task
        
            if remember and (self.learning_type == 'sl' or (self.learning_type == 'ql' and self.joint and not joint_warmup and skill < self.agent.n_skills)): self.agent.memorize(event.copy(), 'sl')
            if env_step < self.steps['env'][self.learning_type]-1: state = np.copy(next_state)
            if end_step or (env_step+1) >= max_env_step: break 
            if self.render and ((env_step+1)%10) == 0: self.envs[self.learning_type][self.task].render()
            if learn and self.learning_type == 'ql' and self.joint and not joint_warmup: self.agent.learn_skills()
             
        if self.learning_type in ['ql', 'tl']:
            masked_done = float(done) if not self.masked_done else float(end_step)
            event = np.empty(2*self.s_dim+4) if self.learning_type == 'ql' else np.empty(2*self.s_dim+5)            
            event[:self.s_dim] = initial_state 
            event[self.s_dim] = skill
            event[self.s_dim+1] = total_reward
            event[self.s_dim+2:2*self.s_dim+2] = final_state
            event[2*self.s_dim+2] = masked_done
            event[2*self.s_dim+3] = task
            if self.learning_type == 'tl': event[2*self.s_dim+4] = concept
            if remember:
                if (self.learning_type == 'ql' and (not self.joint or self.agent.stoA_learning_type == 'SAC')) or self.learning_type == 'tl': self.agent.memorize(event.copy(), self.learning_type)

        if learn:
            if self.learning_type == 'sl':
                for _ in range(0, self.steps['grad']):
                    self.agent.learn_skills()
            elif self.learning_type == 'ql':
                for _ in range(0, self.steps['grad']):
                    if not self.joint or self.agent.stoA_learning_type == 'SAC':
                        self.agent.learn_DQN()                    
            elif self.learning_type == 'tl':
                for _ in range(0, self.steps['grad']):
                    self.agent.learn_transfer_policy(self.learning_type)                    

        return total_reward, done, event, env_step+1, skill 

    def train_agent(self, initialization=True, skill_learning=True, storing_path='', rewards=[], metrics=[], losses=[], entropies=[], entropies_2=[], 
                    iter_0=0, q_learning=True, concept_learning=True, transfer_learning=True):
        if len(storing_path) == 0: storing_path = self.params['storing_path']

        if initialization:
            self.initialization()
            specific_path = storing_path + '/' + str(0)
            self.save(storing_path, specific_path)
        
        init_iter = iter_0
        if skill_learning and not self.joint:
            self.train_agent_skills(storing_path=storing_path, rewards=rewards, metrics=metrics, iter_0=init_iter)
            init_iter = 0
        
        if q_learning:
            self.agent.memory['sl'].forget()
            if not self.joint:
                self.learning_type = 'ql'                
                self.set_envs()
                self.train_agent_skills(storing_path=storing_path, iter_0=init_iter)                
            else:
                for i in range(0, self.joint_cycles):
                    self.agent.restart_policies()
                    self.train_agent_skills(storing_path=storing_path, iter_0=0, joint_warmup=True)                    
                    self.train_agent_skills(storing_path=storing_path, iter_0=init_iter)
                    init_iter += self.epsds['tr'][self.learning_type] // self.epsds['eval']['interval']
            init_iter = 0

        self.learning_type = 'cl'
        if concept_learning:
            self.train_agent_concepts(storing_path=storing_path, iter_0=init_iter, losses=losses, entropies=entropies, entropies_2=entropies_2)
            init_iter = 0
        
        if transfer_learning:
            self.agent.memory['ql'].forget()
            self.learning_type = 'tl'
            self.set_envs()
            self.agent.classifier.eval()
            self.train_agent_skills(storing_path=storing_path, iter_0=init_iter)
    
    @property
    def keep_track(self):
        return (self.active_RND and self.learning_type == 'tl') or (self.active_MC and self.learning_type == 'tl')

    def train_agent_skills(self, iter_0=0, rewards=[], metrics=[], lengths=[], storing_path='', joint_warmup=False):        
        if self.render: self.envs[self.learning_type][self.task].render()   
        
        lim_epsd = self.epsds['tr'][self.learning_type] if not joint_warmup else self.epsds['tr']['wu']
        for epsd in range(0, lim_epsd):
            change_env = False if epsd == 0 else True
            self.reset(change_env=change_env)
            iter_ = iter_0 + (epsd+1) // self.epsds['eval']['interval']
            step_counter = 0
            previous_skill = self.task
            if self.keep_track: 
                trajectory = []
                if self.active_MC: trajectory_MC = []
            
            for epsd_step in itertools.count(0):
                learn = epsd != 0 or epsd_step+1 > 3*self.batch_size or self.joint
                if self.learning_type == 'tl':
                    learn = learn and ((step_counter + 1) % self.steps['update']['tl']) == 0
                done, event, env_steps, previous_skill = self.interaction(learn=learn, lim=self.steps['tr'][self.learning_type]-step_counter, previous_skill=previous_skill, joint_warmup=joint_warmup)[1:]

                if self.render: self.envs[self.learning_type][self.task].render()
                if self.keep_track: trajectory.append(event.copy())
                if self.keep_track and self.active_MC: 
                    S, A, R = event[2*self.s_dim+4], event[self.s_dim], event[self.s_dim+1]
                    trajectory_MC.append([S,A,R])

                if done: 
                    self.reset(change_env=False)
                    if self.keep_track:
                        if self.active_MC:
                            self.agent.MC_learning(trajectory_MC)
                            trajectory_MC = []                              
                step_counter += env_steps

                if (epsd_step + 1) % (512) == 0 and self.agent.active_intrinsic_learning: # TODO: add 512 to param dict.
                    self.agent.intrinsic_learning(trajectory)
                    trajectory = []

                if step_counter >= self.steps['tr'][self.learning_type] * self.steps['env'][self.learning_type]: 
                    if self.keep_track:                    
                        if len(trajectory) >= 1:
                            if self.active_RND and self.agent.active_intrinsic_learning:
                                self.agent.intrinsic_learning(trajectory)
                            trajectory = []
                            if self.active_MC:
                                self.agent.MC_learning(trajectory_MC)
                                trajectory_MC = []
                    break
                    
                if self.multitask_envs[self.learning_type] and ((step_counter+1) % self.steps['MT']) == 0: 
                    self.MT_task = (self.MT_task + np.random.randint(self.n_MT_tasks-1) + 1) % self.n_MT_tasks                
            
            if (epsd+1) % self.epsds['eval']['interval'] == 0 and not joint_warmup:
                st0_random = random.getstate()
                st0 = np.random.get_state() 
                st0_rng = self.np_random.get_state()
                st0_torch = torch.get_rng_state() 
                if device == "cuda": st0_torch_cuda = torch.cuda.get_rng_state()  
                st_envs = []
                for env in self.envs[self.learning_type]:
                    st_envs.append(env.np_random.get_state())
                                 
                r, _, m, l = self.eval_agent_skills(explore=(self.learning_type=='sl'), iter_=iter_, store_events=False)
                random.setstate(st0_random)
                np.random.set_state(st0)
                self.np_random.set_state(st0_rng)
                torch.set_rng_state(st0_torch)
                if device == "cuda": torch.cuda.set_rng_state(st0_torch_cuda)
                for i, env in enumerate(self.envs[self.learning_type]):
                    env.np_random.set_state(st_envs[i])

                metrics.append(m)
                rewards += r
                if self.learning_type == 'tl': 
                    lengths.append(l)
                    np.savetxt(storing_path + '/lengths_'+self.learning_type+'.txt', np.array(lengths))
                np.savetxt(storing_path + '/metrics_'+self.learning_type+'.txt', np.array(metrics))               
                
                specific_path = storing_path + '/' + str(iter_)
                self.save(storing_path, specific_path=specific_path)
                np.savetxt(storing_path + '/mean_rewards_'+self.learning_type+'.txt', np.array(rewards))
    
    def train_agent_concepts(self, losses=[], entropies=[], entropies_2=[], storing_path='', iter_0=0, min_lr=1e-4, max_lr=3e-4, T=50000, max_tau=5.0, min_tau=1.3, last_steps=50000, max_max_tau=10.0):
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()        

        for grad_step in range(0, self.steps['tr'][self.learning_type]):
            classifier_loss, HS_T,  HS_s, ISs_T, ILBO_AS_T_term1, HA_T, ILBO_AS_T, ILBO_nSS_TdoA_term1, HnS_TdoA, ILBO_nSS_TdoA, cl, ml = self.agent.learn_concepts()
            losses.append([classifier_loss, cl, ml])
            entropies.append([HS_T,  HS_s, ISs_T, ILBO_AS_T_term1, HA_T, ILBO_AS_T, ILBO_nSS_TdoA_term1, HnS_TdoA, ILBO_nSS_TdoA])

            stdscr.addstr(0, 0, "Iteration: {}".format(grad_step))
            stdscr.addstr(1, 0, "Classifier Loss: {}".format(np.round(classifier_loss, 4)))
            stdscr.addstr(2, 0, "Entropy H(S|T): {}".format(np.round(HS_T, 4)))
            stdscr.addstr(3, 0, "Entropy H(S|s): {}".format(np.round(HS_s,4)))
            stdscr.addstr(4, 0, "MutualInfo I(S:s|T): {}".format(np.round(ISs_T,4)))
            stdscr.addstr(5, 0, "Entropy H(A|S,T): {}".format(np.round(ILBO_AS_T_term1,4)))
            stdscr.addstr(6, 0, "Entropy H(A|T): {}".format(np.round(HA_T,4)))
            stdscr.addstr(7, 0, "MutualInfo I(A:S|T): {}".format(np.round(ILBO_AS_T,4)))
            stdscr.addstr(8, 0, "Entropy H(S'|S,T,do(A)) : {}".format(np.round(ILBO_nSS_TdoA_term1,4)))
            stdscr.addstr(9, 0, "Entropy H(S'|T,do(A)): {}".format(np.round(HnS_TdoA,4)))
            stdscr.addstr(10, 0, "MutualInfo I(S':S|T,do(A)): {}".format(np.round(ILBO_nSS_TdoA,4)))
            stdscr.addstr(11, 0, "Policy model loss: {}".format(np.round(cl, 4)))
            stdscr.addstr(12, 0, "Transition model loss: {}".format(np.round(ml, 4)))
            stdscr.refresh()

            if (grad_step + 1) % 5000 == 0:
                self.save(storing_path, storing_path+ '/' + str(iter_0+grad_step+1))
                np.savetxt(storing_path + '/concept_training_losses.txt', np.array(losses))
                np.savetxt(storing_path + '/concept_training_entropies.txt', np.array(entropies)) 
            
        curses.echo()
        curses.nocbreak()
        curses.endwin()         

    @property
    def entropy_metric(self):
        return self.learning_type == 'sl' or self.agent.stoA_learning_type == 'SAC'

    def eval_agent_skills(self, eval_epsds=0, explore=False, iter_=0, start_render=False, print_space=True, specific_path='video', max_step=0, task=None, store_events=True):
        if task is None:   
            task = self.task
            self.task = 0
            given_task = False
            if self.multitask_envs[self.learning_type]:
                MT_task = self.MT_task
                self.MT_task = 0
        else:
            self.task = task
            given_task = True
        self.reset()

        if start_render: self.envs[self.learning_type][self.task].render()
        if eval_epsds == 0: 
            if self.multitask_envs[self.learning_type]:
                eval_epsds = self.epsds['eval'][self.learning_type] * self.n_MT_tasks
            else:
                eval_epsds = self.epsds['eval'][self.learning_type] * self.n_tasks[self.learning_type]
        
        events = []
        rewards = []
        epsd_lengths = []
        min_epsd_reward = 1.0e6
        max_epsd_reward = -1.0e6

        if self.entropy_metric:
            Ha_sT = []
            Ha_sT_average = 0.0
            entropy = 'H(a|s,T)' if self.learning_type == 'sl' else 'H(A|s,T)'
        
        if max_step <= 0: max_step = self.steps['eval'][self.learning_type]

        for epsd in range(0, eval_epsds):
            step_counter = 0

            if self.store_video: video = VideoWriter(specific_path + '_' + str(self.task) + '_' + str(epsd) + '_' + self.learning_type + '.avi', fourcc, float(FPS), (width, height))

            change_env = False if epsd == 0 or given_task else True
            self.reset(change_env=change_env)           
            epsd_reward = 0.0
            previous_skill = self.task

            for eval_step in itertools.count(0):            
                reward, done, event, env_steps, previous_skill = self.interaction(remember=False, explore=explore, learn=False, lim=self.steps['tr'][self.learning_type]-step_counter, previous_skill=previous_skill)
                if self.learning_type == 'sl':
                    event[self.sa_dim] = reward  
                epsd_reward += reward  
                if self.learning_type == 'tl':
                    step_counter += env_steps                            

                if self.store_video:
                    img = self.envs[self.learning_type][self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif self.render:
                    self.envs[self.learning_type][self.task].render()

                if store_events:
                    if self.env_names[self.learning_type][self.task] == 'AntCrossMaze-v3':
                        goal_position = np.copy(self.envs[self.learning_type][self.task]._goal_position[:2])
                        event = np.concatenate([event, goal_position])
                    if self.env_names[self.learning_type][self.task] in ['AntGather-v3', 'AntAvoid-v3']:
                        object_positions = np.copy(self.envs[self.learning_type][self.task]._object_positions[:,:2].reshape(-1))
                        event = np.concatenate([event, object_positions])
                    events.append(event)

                if done or ((eval_step + 1) >= max_step):
                    if self.learning_type != 'tl':
                        epsd_lengths.append(eval_step + 1)
                    else:
                        epsd_lengths.append(step_counter)
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
                stdout.write("Iter %i, epsd %i, min r: %.3f, max r: %.3f, mean r: %.3f, epsd r: %.3f\r " %
                    (iter_, (epsd+1), min_epsd_reward, max_epsd_reward, average_reward, epsd_reward))
            stdout.flush()  

            self.MT_task = (self.MT_task + 1) % self.n_MT_tasks
            # self.MT_task = (self.MT_task + np.random.randint(self.n_MT_tasks-1) + 1) % self.n_MT_tasks   

        if print_space: print("")

        if self.store_video: video.release()
        metric_vector = np.array([Ha_sT_average]) if self.entropy_metric else np.array([]) 
        
        if not given_task: 
            self.task = task
            if self.multitask_envs[self.learning_type]:
                self.MT_task = MT_task
        return rewards, np.array(events), metric_vector, np.array(epsd_lengths)      
    
    def save(self, common_path, specific_path=''):
        self.params['learning_type'] = self.learning_type
        pickle.dump(self.params, open(common_path+'/params.p','wb'))
        self.agent.save(common_path, specific_path, self.learning_type)
        if self.joint: self.agent.save(common_path, specific_path, 'sl')
    
    def load(self, common_path, iter_0_sl=0, iter_0_sl_2=0, iter_0_ql=0, iter_0_cl=0, iter_0_tl=0, load_memory=True):  # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        if not self.joint:
            if iter_0_sl > 0:
                self.agent.load(common_path, common_path + '/' + str(iter_0_sl), 'sl', load_memory=(load_memory and iter_0_ql==0))
            if iter_0_sl_2 > 0:
                self.agent.load(common_path, common_path + '/' + str(iter_0_sl_2), 'sl_2', load_memory=False)
            if iter_0_ql > 0:
                self.agent.load(common_path, common_path + '/' + str(iter_0_ql), 'ql', load_memory=(load_memory and iter_0_tl==0))
            if iter_0_cl > 0:
                self.agent.load(common_path, common_path + '/' + str(iter_0_cl), 'cl')
            if iter_0_tl > 0:
                self.agent.load(common_path, common_path + '/' + str(iter_0_tl), 'tl', load_memory=load_memory)
        else:
            if iter_0_ql > 0:
                self.agent.load(common_path, common_path + '/' + str(iter_0_ql), 'sl', load_memory=load_memory)
                self.agent.load(common_path, common_path + '/' + str(iter_0_ql), 'ql', load_memory=False)

        

