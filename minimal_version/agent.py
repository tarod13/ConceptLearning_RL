import numpy as np
import random

import torch
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from nets_seq import (Memory, v_Net, q_Net, NoisyDuelingDQN, DuelingDQN, s_Net, c_Net, RND_Net, d_Net, HeapPriorityQueue,
                        discrete_AC_Net_PG, discrete_AC_Net_PG_simple, DQN_actor_Net)

# If Prioritized Experience Replay
from memory_ku2482 import LazyPrioritizedMemory
from utils_ku2482 import to_batch

from utils import updateNet, is_float, is_tensor

import os
import time
import pickle
from sys import stdout
import itertools
import curses

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

#-------------------------------------------------------------
#
#                          Agent class
#
#-------------------------------------------------------------
class Agent:
    def __init__(self, s_dim, a_dim, n_tasks, params, seed=0, joint=False):

        self.params = params.copy()
        default_params = {
                            'n_concepts': 10,
                            'n_skills': 8,
                            'decision_type': 'eps',
                            'tl_type': 'CGSAC',
                            'alpha': {
                                        'sl': 0.1,
                                        'ql': 0.1,
                                        'cl': 1e-6,                                        
                                    },
                            'init_beta': {
                                            'ql': 1e-1,
                                            'tl': 1e-1
                                        },
                            'init_eta': {
                                            'ql': 1e-2,
                                            'tl': 1e-2
                                        },
                            'init_epsilon': 1.0,
                            'min_epsilon': 0.4,                            
                            'delta_epsilon': 2.5e-7,                            
                            'init_threshold_entropy_alpha': 0.0,
                            'init_threshold_entropy_alpha_cl': np.log(10),
                            'delta_threshold_entropy_alpha': 8e-6,
                            'delta_threshold_entropy_alpha_cl': 3.2e-6,
                            'min_threshold_entropy_alpha_ql': np.log(2),
                            'min_threshold_entropy_alpha_cl': np.log(2),
                            'stoA_learning_type': 'SAC',
                            'DQL_epsds_target_update': 6000,                            
                            
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
                                            'v_target': 3e-4,
                                            'target': 5e-3,
                                            'beta': 3e-4,
                                        },
                                    'cl': {
                                            'alpha': 3e-4,
                                            'c': 3e-5
                                        },
                                    'tl': {
                                            'target': 5e-3,
                                            'beta': 3e-4,
                                            'eta': 3e-4
                                        }
                                    },

                            'dims': {
                                        'init_prop': 2,
                                        'last_prop': s_dim,
                                        'init_ext': 3,
                                        'last_ext': s_dim-60
                                    },
                            
                            'batch_size': {
                                            'sl': 256,
                                            'ql': 256,
                                            'tl': 256
                                        },
                            'RND_factor': 1.0,
                            'memory_capacity': 1200000,
                            'GAE_lambda': 0.95,
                            'gamma_E': 0.99,
                            'gamma_I': 0.975,                            
                            'clip_value': 0.5,
                            'classification_with_entropies': False,
                            'n_update_cycles_ps': 10,
                            'nu': 1.0,
                            'n_steps_classifier_model': 1,
                            'factor_I': 1.0,
                            'entropy_annealing': False,
                            'max_concept_divergence': 0.5*np.log(4),
                            'max_concept_entropy': np.log(2.5),
                            'delta_skill_entropy': 1e-5,
                            'min_min_skill_entropy': np.log(1.05),
                            'surgery': False,
                            'RND_update_proportion': 0.25,
                            'per': False,
                            'alpha_per': 0.6,
                            'beta_per': 0.4,
                            'beta_annealing_per': 0.001,
                            'multi_step_per': 1,
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

        self.joint = joint
        self.n_tasks = n_tasks
        self.seed = seed
        self.n_skills = n_tasks['sl'] if not joint else self.params['n_skills']
        self.counter = {
                        'sl': 0,
                        'ql': 0
                    }
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
        self.RND_factor = self.params['RND_factor']
        self.classification_with_entropies = self.params['classification_with_entropies']
        self.n_update_cycles_ps = self.params['n_update_cycles_ps']
        self.n_steps_classifier_model = self.params['n_steps_classifier_model']
        self.entropy_annealing = self.params['entropy_annealing']
        self.tl_type = self.params['tl_type']
        self.surgery = self.params['surgery']
        self.RND_update_proportion = self.params['RND_update_proportion']
        self.per = self.params['per']
        self.active_intrinsic_learning = self.params['intrinsic_learning']

        self.max_concept_divergence = self.params['max_concept_divergence']
        self.max_concept_entropy = self.params['max_concept_entropy']
        self.min_skill_entropy = 0.95 * (-np.log(1/(self.n_skills+1)))  #np.log((self.n_skills+1)*0.75) if not self.joint else np.log((self.n_skills)*0.75)
        self.min_min_skill_entropy = self.params['min_skill_entropy']
        self.delta_skill_entropy = self.params['delta_skill_entropy']
        self.max_noise = 1e-10
        self.r_mean = 0.0
        self.r_std = 1.0
        self.r_initialized = False
        self.r_max = 0.0
        self.novelty_factor = 1.0/np.log(100.0)
        self.novelty_factor_small = 1.0/np.log(20.0)

        # Metric weights
        self.min_threshold_entropy_alpha = {
                                            'sl': -a_dim*1.0/2,
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
        for learning_type in ['sl', 'ql', 'cl']:
            self.alpha[learning_type] = alpha[learning_type]
        # for learning_type in ['sl']:
        #     self.alpha[learning_type] = (alpha[learning_type] * torch.ones(v_dim).float().to(device) if is_float(alpha[learning_type]) else 
        #                     (alpha[learning_type].float().to(device) if is_tensor(alpha[learning_type]) else torch.from_numpy(alpha[learning_type]).float().to(device)))
        self.beta = {}
        beta = self.params['init_beta']
        for learning_type in ['ql', 'tl']:
            self.beta[learning_type] = (beta[learning_type] * torch.ones(self.n_tasks[learning_type]).float().to(device) if is_float(beta[learning_type]) else 
                                        (beta[learning_type].float().to(device) if is_tensor(beta[learning_type]) else torch.from_numpy(beta[learning_type]).float().to(device)))
        self.eta = {}
        eta = self.params['init_eta']
        for learning_type in ['ql', 'tl']:
            self.eta[learning_type] = (eta[learning_type] * torch.ones(self.n_tasks[learning_type]).float().to(device) if is_float(eta[learning_type]) else 
                                        (eta[learning_type].float().to(device) if is_tensor(eta[learning_type]) else torch.from_numpy(eta[learning_type]).float().to(device)))
        
        self.epsilon = self.params['init_epsilon']
        self.min_epsilon = self.params['min_epsilon']
        self.delta_epsilon = self.params['delta_epsilon']
        self.nu = self.params['nu']
        self.factor_I = self.params['factor_I']

        self.lambda_H = 1e-2
        self.lim_HSs = np.log(20)        
              
       # Nets and memory
        self.v = {
                            'sl': v_Net(self.dims['last_ext']-self.dims['init_ext'], v_dim, lr=self.lr['sl']['v']).to(device),                            
                        }
        self.v_target = {
                            'sl': v_Net(self.dims['last_ext']-self.dims['init_ext'], v_dim, lr=self.lr['sl']['v']).to(device),                            
                        }
        if self.stoA_learning_type == 'DQL':
            self.critic1 = {
                                'sl': q_Net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr['sl']['q']).to(device),
                                'ql': DuelingDQN(self.dims['last_ext']-self.dims['init_ext'], self.n_skills+1, n_tasks['ql'], lr=self.lr['ql']['q']).to(device),
                            }
            self.critic2 = {
                                'sl': q_Net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr['sl']['q']).to(device),
                                'ql': DuelingDQN(self.dims['last_ext']-self.dims['init_ext'], self.n_skills+1, n_tasks['ql'], lr=self.lr['ql']['q']).to(device),
                            }
        else:
            self.critic1 = {
                                'sl': q_Net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr['sl']['q']).to(device),
                                'ql': discrete_AC_Net_PG_simple(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], n_tasks['ql'], lr=self.lr['ql']['q']).to(device),
                            }
            self.critic2 = {
                                'sl': q_Net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr['sl']['q']).to(device),
                            }

        self.actor = s_Net(self.n_skills, self.dims['last_prop']-self.dims['init_prop'], a_dim, lr=self.lr['sl']['pi']).to(device)
        self.classifier = c_Net(self.n_concepts, self.dims['last_ext']-self.dims['init_ext'], self.n_skills+1, n_tasks=self.n_tasks['ql'], lr=3.0e-4).to(device)
        
        self.memory = {
                        'sl':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'ql':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        # 'tl':  Memory(self.params['memory_capacity'], n_seed=self.seed)
                    }
        
        if self.per:
            # replay memory with prioritied experience replay
            self.memory['tl'] = LazyPrioritizedMemory(
                self.params['memory_capacity'], (self.s_dim,),
                device, self.gamma_E, self.params['multi_step_per'],
                alpha=self.params['alpha_per'], beta=self.params['beta_per'], beta_annealing=self.params['beta_annealing_per'])
        else:
            self.memory['tl'] = Memory(self.params['memory_capacity'], n_seed=self.seed)

        # self.PS_T = torch.ones(self.n_tasks['ql'], self.n_concepts).to(device) / self.n_concepts
        # self.PA_ST = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1).to(device) / (self.n_skills+1)
        # self.PnS_STdoA = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1, self.n_concepts).to(device) / self.n_concepts
        self.NnSdoAST_cl = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1, self.n_concepts).to(device)
        self.NAST_cl = torch.ones(self.n_tasks['ql'], self.n_concepts, self.n_skills+1).to(device)
        
        self.RND = {
                        'sl': RND_Net(self.dims['last_ext']-self.dims['init_ext'], self.n_skills).to(device),
                        'ql': RND_Net(self.dims['last_ext']-self.dims['init_ext'], self.n_tasks['ql']).to(device),
                        'tl': RND_Net(self.dims['last_ext']-self.dims['init_ext'], self.n_tasks['tl'], gamma_I=self.gamma_I).to(device) # TODO TODO TODO TODO: fix when n_tasks['tl'] is 0
                    }
        
        self.NAST = torch.ones(self.n_tasks['tl'], self.n_concepts, self.n_skills+1).to(device)
        self.PA_ST_tl = torch.ones(self.n_concepts, self.n_skills+1).to(device)
        self.QAST = torch.ones(self.n_concepts, self.n_skills+1).to(device)

        self.NAST_MC = torch.zeros(self.n_concepts, self.n_skills+1).to(device)
        self.QAST_MC = torch.zeros(self.n_concepts, self.n_skills+1).to(device)

        if self.tl_type in ['CGSAC', 'SAC']:
            self.CG_actor = discrete_AC_Net_PG(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], n_tasks['tl'], self.n_concepts, lr=3e-4).to(device)
        elif self.tl_type == 'DQN':
            self.CG_actor = DQN_actor_Net(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], n_tasks['tl'], lr=3e-4).to(device)
        else:
            assert 0 == 1, 'Wrong transfer learning type'
        if self.joint:
            self.SAC = discrete_AC_Net_PG(self.n_skills, self.dims['last_ext']-self.dims['init_ext'], n_tasks['ql'], lr=3e-3).to(device)
        else:
            self.SAC = discrete_AC_Net_PG(self.n_skills+1, self.dims['last_ext']-self.dims['init_ext'], n_tasks['tl'], self.n_concepts, lr=3e-3).to(device)
        self.PA_ST_SAC = torch.ones(self.n_tasks['tl'], self.n_concepts, self.n_skills+1).to(device) / (self.n_skills+1)
        self.credibility = 1e-10*torch.ones(self.n_tasks['tl'], self.n_concepts).to(device)
        
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
            return self.classifier.sample_concept(state_cuda, explore=explore)[0]

    def decide(self, state, task, learning_type, explore=True, guess=False, rng=None):
        with torch.no_grad():
            if learning_type == 'ql':
                if not self.joint:
                    if self.stoA_learning_type == 'DQL':
                        skill = self.decide_q_dist(state, task, explore=explore) if self.decision_type == 'q_dist' else self.decide_epsilon(state, task, explore=explore)
                    else:
                        s_cuda = torch.FloatTensor(state[self.dims['init_ext']:self.dims['last_ext']]).to(device).view(1,-1)
                        skill = self.critic1['ql'].sample_skill(s_cuda, task, explore=explore, rng=rng)
                else:
                    s_cuda = torch.FloatTensor(state[self.dims['init_ext']:self.dims['last_ext']]).to(device).view(1,-1)
                    skill = self.SAC.sample_skill(s_cuda, task, explore=explore)
                return skill 
            elif learning_type == 'tl':
                s_cuda = torch.FloatTensor(state[self.dims['init_ext']:self.dims['last_ext']]).to(device).view(1,-1)
                if self.tl_type == 'SAC':
                    skill = self.SAC.sample_skill(s_cuda, task, explore=explore)
                else:
                    skill = self.CG_actor.sample_skill(s_cuda, task, explore=explore, rng=rng)
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
            # qe, qi_exp = self.critic1['ql'](s_cuda)
            # qe, qi_exp = qe.squeeze(0)[task,:], qi_exp.squeeze(0)[task,:]
            qe = self.critic1['ql'](s_cuda)
            qe = qe.squeeze(0)[task,:]
            # epsilon = self.epsilon if explore else 0.0
            # skill = (qe+qi_exp).argmax().item() if np.random.rand() > epsilon else np.random.randint(self.n_skills+1)
            tie_breaking_dist = torch.isclose(qe, qe.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            skill = Categorical(probs=tie_breaking_dist).sample().cpu() 
            skill = skill if np.random.rand() > self.epsilon else np.random.randint(self.n_skills+1)
            return skill            

    def act(self, state, skill, explore=True, learning_type='sl'):
        # if learning_type == 'sl':
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

    def learn_DQN_DQL(self, learning_type='ql'):
        self.counter['ql'] += 1
        # batch, indices, priorities = self.memory['ql'].sample(self.batch_size['ql'])
        batch = self.memory[learning_type].sample(self.batch_size[learning_type])
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
            # qe, qi_exp = self.critic1['ql'](s_batch)
            # qe, qi_exp = qe[np.arange(batch_size), T_batch, A_batch].view(-1,1), qi_exp[np.arange(batch_size), T_batch, A_batch].view(-1,1)
            # qen, qi_expn = self.critic1['ql'](ns_batch)
            # qen, qi_expn = qen[np.arange(batch_size), T_batch, :], qi_expn[np.arange(batch_size), T_batch, :]
            # qen_target, qi_expn_target =  self.critic2['ql'](ns_batch)
            # qen_target, qi_expn_target =  qen_target[np.arange(batch_size), T_batch, :], qi_expn_target[np.arange(batch_size), T_batch, :]
            if learning_type == 'ql':
                qe = self.critic1['ql'](s)[np.arange(batch_size), T, A].view(-1,1)
                nqe = self.critic1['ql'](ns)[np.arange(batch_size), T, :]
                nqe_target =  self.critic2['ql'](ns)[np.arange(batch_size), T, :]
                
                # RND_error = self.RND['ql'](ns_batch, T_batch)
                
                # best_skills = (qen+qi_expn).argmax(1)
                best_skills = nqe.argmax(1)
                qe_approx = re/10.0 + self.gamma_E * nqe_target[np.arange(batch_size), best_skills].view(-1,1) * (1.0-d) # +  0.5*RND_error.detach()
                # qi_exp_approx = RND_error.detach() + self.gamma_I * qi_expn_target[np.arange(batch_size), best_skills].view(-1,1) * (1.0-d_batch)

                # new_sampling_priorities = list((((qe + qi_exp - qe_approx - qi_exp_approx).squeeze(1).abs()+1e-2)**0.6).detach().cpu().numpy())

                q_loss = self.critic1['ql'].loss_func(qe, qe_approx.detach())# + self.critic1['ql'].loss_func(qi_exp, qi_exp_approx.detach()))*IS_weights
                self.critic1['ql'].optimizer.zero_grad()
                q_loss.mean().backward()
                clip_grad_norm_(self.critic1['ql'].parameters(), self.clip_value)
                self.critic1['ql'].optimizer.step()

                # self.RND['ql'].optimizer.zero_grad()
                # RND_error.mean().backward()
                # clip_grad_norm_(self.RND['ql'].predictor.parameters(), self.clip_value)
                # self.RND['ql'].optimizer.step()
                
                # self.memory['ql'].update_weights(indices, new_sampling_priorities)

                if self.counter['ql'] % self.DQL_epsds_target_update == 0:
                    updateNet(self.critic2['ql'], self.critic1['ql'], 1.0)
                    self.counter['ql'] = 0

                # Anneal epsilon
                self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])
            else:
                qe, _, qi_exp, _, qi_con, _ = self.CG_actor(s, T)
                qe_A, qi_exp_A, qi_con_A = qe[np.arange(batch_size), A].view(-1,1), qi_exp[np.arange(batch_size), A].view(-1,1), qi_con[np.arange(batch_size), A].view(-1,1)
                nqe, nqe_target, nqi_exp, nqi_exp_target, nqi_con, nqi_con_target = self.CG_actor(ns, T)
                
                A_off_one_hot = torch.zeros(batch_size, qe.size(1)).float().to(device)
                A_off_one_hot[np.arange(batch_size), qe.argmax(1)] = torch.ones(batch_size).float().to(device)

                T_one_hot = torch.zeros(batch_size, self.n_tasks['tl']).float().to(device)
                T_one_hot[np.arange(batch_size), T] = torch.ones(batch_size).float().to(device)
                nT = T_one_hot.sum(0).view(-1,1)
                T_dist = T_one_hot / (nT.view(1,-1) + 1e-10)
                PT = nT / batch_size

                PS_s, _ = self.classifier.classifier(s)
                NAST_batch = torch.einsum('ij,ihk->hjk', PS_s, A_off_one_hot.unsqueeze(1) * T_one_hot.unsqueeze(2))
                df = 0.01 * (self.NAST.sum(2, keepdim=True) >= 1.0).float()
                NAST = (1-df) * self.NAST + NAST_batch
                PA_ST = NAST / NAST.sum(2, keepdim=True)                
                PST = torch.einsum('ij,ih->hj', PS_s, T_dist).detach() + 1e-10
                PS_T = PST / PST.sum(1, keepdim=True)
                
                log_PA_ST = torch.log(PA_ST + 1e-10) 
                HA_ST = -(PA_ST * log_PA_ST).sum(2)

                ri_exploration = self.RND[learning_type](ns, T) / 5.0
                confidence = 1-torch.exp(-NAST.sum(2) * 5e-4)
                ri_consensus = (confidence[T, :] * PS_s * PA_ST[T, :, A]).sum(1, keepdim=True)
                # ri = ri_exploration + ri_consensus

                best_nskills = (nqe + nqi_exp + nqi_con).argmax(1)
                qe_approx = re + self.gamma_E * nqe_target[np.arange(batch_size), best_nskills].view(-1,1) * (1.0-d)
                qi_exp_approx = ri_exploration.detach() + self.gamma_I * nqi_exp_target[np.arange(batch_size), best_nskills].view(-1,1) # * (1.0-d)
                qi_con_approx = ri_consensus.detach() + self.gamma_I * nqi_con_target[np.arange(batch_size), best_nskills].view(-1,1) # * (1.0-d)

                qe_loss = ((qe_A - qe_approx.detach())**2).sum(1, keepdim=True)
                self.CG_actor.qe.optimizer.zero_grad()
                qe_loss.mean().backward()
                clip_grad_norm_(self.CG_actor.qe.parameters(), self.clip_value)
                self.CG_actor.qe.optimizer.step()

                qi_exp_loss = ((qi_exp_A - qi_exp_approx.detach())**2).sum(1, keepdim=True)
                self.CG_actor.qi_exploration.optimizer.zero_grad()
                qi_exp_loss.mean().backward()
                clip_grad_norm_(self.CG_actor.qi_exploration.parameters(), self.clip_value)
                self.CG_actor.qi_exploration.optimizer.step()

                qi_con_loss = ((qi_con_A - qi_con_approx.detach())**2).sum(1, keepdim=True)
                self.CG_actor.qi_consensus.optimizer.zero_grad()
                qi_con_loss.mean().backward()
                clip_grad_norm_(self.CG_actor.qi_consensus.parameters(), self.clip_value)
                self.CG_actor.qi_consensus.optimizer.step()

                self.RND[learning_type].optimizer.zero_grad()
                ri_exploration.mean().backward()
                clip_grad_norm_(self.RND[learning_type].predictor.parameters(), self.clip_value)
                self.RND[learning_type].optimizer.step()

                if self.counter['ql'] % self.DQL_epsds_target_update == 0:
                    self.CG_actor.update_targets(1.0)
                    self.counter['ql'] = 0
                
                self.NAST = NAST.detach().clone()
                
    def learn_DQN_SAC(self, learning_type, only_metrics=False, learn_alpha=True):
        batch = self.memory[learning_type].sample(self.batch_size[learning_type])
        batch = np.array(batch)
        batch_size = batch.shape[0]
        weights = 1.                  
        
        if batch_size > 0:
            self.counter['ql'] += 1
            s = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            A = batch[:,self.s_dim].astype('int')
            re = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device) / 10.0
            ns = torch.FloatTensor(batch[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            d = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
            T = batch[:,2*self.s_dim+3].astype('int')            
            
            # Optimize Q
            qe1, _, qe2, _, PA_sT, log_PA_sT, alpha, log_alpha = self.critic1['ql'](s, T)
            _, nqe1t, _, nqe2t, PnA_nsT, log_PnA_nsT, nalpha, log_nalpha = self.critic1['ql'](ns, T)
            
            nqet = torch.min(nqe1t, nqe2t)
            nve = (PnA_nsT * (nqet - alpha.view(-1,1) * log_PnA_nsT)).sum(1, keepdim=True).detach()
            qe_approx = re + self.gamma_E * nve * (1-d)

            qe1_A, qe2_A = qe1[np.arange(batch_size), A].view(-1,1), qe2[np.arange(batch_size), A].view(-1,1)
            qe1_loss = (qe1_A - qe_approx.detach())**2
            qe2_loss = (qe2_A - qe_approx.detach())**2
            errors = torch.max(qe1_loss, qe2_loss).detach()**0.5
            
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
                alpha_loss = log_alpha * (H_pi_target - scaled_min_entropy).detach() # .clamp(-20,2.0)

            if not only_metrics:
                self.critic1['ql'].qe1.optimizer.zero_grad()
                (qe1_loss * weights).mean().backward()
                clip_grad_norm_(self.critic1['ql'].qe1.parameters(), self.clip_value)
                self.critic1['ql'].qe1.optimizer.step()

                self.critic1['ql'].qe2.optimizer.zero_grad()
                (qe2_loss * weights).mean().backward()
                clip_grad_norm_(self.critic1['ql'].qe2.parameters(), self.clip_value)
                self.critic1['ql'].qe2.optimizer.step()
                
                if learn_alpha:
                    self.critic1['ql'].alpha_optim.zero_grad()
                    (alpha_loss * weights).mean().backward()
                    self.critic1['ql'].alpha_optim.step()
                    self.critic1['ql'].alpha = self.critic1['ql'].log_alpha.exp()

                self.critic1['ql'].actor.optimizer.zero_grad()
                (pi_loss * weights).mean().backward() # retain_graph=CG)
                clip_grad_norm_(self.critic1['ql'].actor.parameters(), self.clip_value)
                self.critic1['ql'].actor.optimizer.step()

                self.critic1['ql'].update_targets(self.lr['tl']['target'])  

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
        if self.tl_type == 'SAC':
            metrics = self.learn_DQN_SAC(learning_type, only_metrics=only_metrics)
            return metrics
        elif self.tl_type == 'CGSAC':
            self.CG_SAC_learning(only_metrics=only_metrics)
        else:
            self.learn_DQN_DQL(learning_type=learning_type)

    def CG_SAC_learning(self, only_metrics=False, CG=True, learn_alpha=True):
        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory['tl'].sample(self.batch_size['tl'])
            batch_size = batch[0].shape[0]
            T = np.zeros(batch_size).astype('int')
        else:
            batch = self.memory['tl'].sample(self.batch_size['tl'])
            batch = np.array(batch)
            batch_size = batch.shape[0]
            # set priority weights to 1 when we don't use PER.
            weights = 1.       
        
        if batch_size > 0:
            self.counter['ql'] += 1
            if self.per:
                s, A, re, ns, d, _ = batch
                A = A.astype('int')
                s, ns = s[:,self.dims['init_ext']:self.dims['last_ext']], ns[:,self.dims['init_ext']:self.dims['last_ext']]
            else:
                s = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
                A = batch[:,self.s_dim].astype('int')
                re = torch.FloatTensor(batch[:,self.s_dim+1]).view(-1,1).to(device)
                ns = torch.FloatTensor(batch[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
                d = torch.FloatTensor(batch[:,2*self.s_dim+2]).view(-1,1).to(device)
                T = batch[:,2*self.s_dim+3].astype('int')  

            T_one_hot = torch.zeros(batch_size, self.n_tasks['tl']).float().to(device)
            T_one_hot[np.arange(batch_size), T] = torch.ones(batch_size).float().to(device)
            nT = T_one_hot.sum(0).view(-1,1)
            T_dist = T_one_hot / (nT.view(1,-1) + 1e-10)

            # self.RND['tl'].update_obs_rms(s, T[0])

            # if not learn_alpha:
            #     alpha = 0.05 * torch.ones(1).to(device)
            # else:
            #     alpha_base = 0.01 * torch.ones(1).to(device)
            #     alpha, log_alpha = self.CG_actor.alpha[T[0]].view(-1,1), self.CG_actor.log_alpha[T[0]].view(-1,1)
            
            # Optimize Q
            qe1, _, qe2, _, qi1_exp, _, qi2_exp, _, qi1_con, qi1_cont, qi2_con, qi2_cont, PA_sT, log_PA_sT, alpha, log_alpha, lambda_, Alpha, log_Alpha = self.CG_actor(s, T)
            _, nqe1t, _, nqe2t, _, _, _, _, nqi1_con, nqi1_cont, nqi2_con, nqi2_cont, PnA_nsT, log_PnA_nsT, nalpha, log_nalpha, nlambda_, _, _ = self.CG_actor(ns, T)
            
            nqet = torch.min(nqe1t, nqe2t)
            # nve = (PnA_nsT * (nqet - nalpha.clamp(0.001, 0.1) * log_PnA_nsT)).sum(1, keepdim=True).detach()
            nve = (PnA_nsT * (nqet - 0.01 * log_PnA_nsT)).sum(1, keepdim=True).detach()
            qe_approx = re + self.gamma_E * nve * (1-d)

            # ri_exp = self.RND['tl'](ns.detach(), T[0]).sum(1, keepdim=True)
            # ri_exp_normalized = ri_exp / (self.RND['tl'].q_rms.var[T[0],:] + 1e-10).detach().item()**0.5
            # ri_exp_normalized = ri_exp / (ri_exp.std() + 1e-10) * (1-self.gamma_I)
            # nqi_expt = torch.min(nqi1_expt, nqi2_expt)
            # nvi_exp = (PnA_nsT * nqi_expt).sum(1, keepdim=True).detach()
            # qi_exp_approx = ri_exp_normalized.detach() + self.gamma_I * nvi_exp # * (1-d)
            # self.RND['tl'].update_q_rms(qi_exp_approx, T[0])
            
            qe1_A, qe2_A = qe1[np.arange(batch_size), A].view(-1,1), qe2[np.arange(batch_size), A].view(-1,1)
            # qi1_exp_A, qi2_exp_A = qi1_exp[np.arange(batch_size), A].view(-1,1), qi2_exp[np.arange(batch_size), A].view(-1,1)
            qe1_loss = (qe1_A - qe_approx.detach())**2
            qe2_loss = (qe2_A - qe_approx.detach())**2
            errors = torch.max(qe1_loss, qe2_loss).detach()**0.5
            # qi1_exp_loss = (qi1_exp_A - qi_exp_approx.detach())**2
            # qi2_exp_loss = (qi2_exp_A - qi_exp_approx.detach())**2
            # if CG:
            #     qi1_con_loss = (qi1_con[np.arange(batch_size), A].view(-1,1) - qi_con_approx.detach())**2
            #     qi2_con_loss = (qi2_con[np.arange(batch_size), A].view(-1,1) - qi_con_approx.detach())**2
           
            PA_T = PA_sT.mean(0, keepdim=True)
            HA_sT = -(PA_sT * log_PA_sT).sum(1, keepdim=True)
            HA_sT_mean = HA_sT.detach().mean()
            # if learn_alpha:
            #     log_novelty_ratios = self.RND['tl'].novelty_ratios(s, T[0])
            #     scaled_min_entropy = self.min_skill_entropy * (1.0 / (1.0 + self.novelty_factor * log_novelty_ratios))
            #     alpha_loss = log_alpha * (HA_sT - scaled_min_entropy).detach()

            # if CG:
            #     qt = torch.min(qe1 + qi1_exp + lambda_ * qi1_con, qe2 + qi2_exp + lambda_ * qi2_con).detach()
            # else:
            qt = torch.min(qe1, qe2).detach() + 1.0*torch.min(qi1_exp, qi2_exp).detach()
            z = torch.logsumexp(qt.detach()/(alpha+1e-10), 1, keepdim=True)

            if CG:
                PS_s = self.classifier.classifier(s)[0].detach()
                # PST = torch.einsum('ij,ih->hj', PS_s, T_dist).detach() + 1e-10
                # PS_T = PST / PST.sum(1, keepdim=True)      

                # qe = torch.min(qe1, qe2)
                # ze = torch.logsumexp(qe/(alpha.detach() + 1e-10), 1, keepdim=True).detach()
                # ideal_PA_sT = torch.exp(qe/(alpha.detach() + 1e-10) - ze)
                ideal_PA_sT = torch.exp(qt/(alpha.detach() + 1e-10) - z).detach()
                # ze = torch.logsumexp(qe/0.01, 1, keepdim=True).detach()
                # ideal_PA_sT = torch.exp(qe/0.01 - ze)
                ideal_PA_sT = ideal_PA_sT / ideal_PA_sT.sum(1, keepdim=True)
                NAST_batch = torch.einsum('ij,ihk->hjk', PS_s, ideal_PA_sT.unsqueeze(1) * T_one_hot.unsqueeze(2))
                df = 0.05 * (self.NAST.sum(2, keepdim=True) >= 1.0).float()
                NAST = (1-df) * self.NAST + NAST_batch # (1-df)
                PS_T = NAST.sum(-1)
                PS_T = PS_T / PS_T.sum(-1, keepdim=True)
                QAST_batch = ((PS_s.unsqueeze(2) * qt.unsqueeze(1)).mean(0) / (PS_T.view(-1,1) + 1e-6)).detach()
                PA_ST_batch = ((PS_s.unsqueeze(2) * ideal_PA_sT.unsqueeze(1)).mean(0) / (PS_T.view(-1,1) + 1e-6)).detach()
                PA_ST_batch = (PA_ST_batch + 1e-10) / ((PA_ST_batch + 1e-10).sum(1, keepdim=True))
                QAST = (1-df[0,:,:]) * self.QAST + df[0,:,:] * QAST_batch
                # PA_ST = (1-df[0,:,:]) * self.PA_ST_tl + df[0,:,:] * PA_ST_batch
                # PA_ST = NAST / NAST.sum(2, keepdim=True)
                # Z = torch.logsumexp(QAST / (alpha[0] + 1e-10), 1, keepdim=True)
                # PA_ST = torch.exp(QAST / (alpha[0] + 1e-10) - Z)
                # PA_ST = (PA_ST + 1e-10) / ((PA_ST + 1e-10).sum(1, keepdim=True))
                Z = torch.logsumexp(self.QAST_MC.detach() / (Alpha.view(-1,1) + 1e-10), 1, keepdim=True)
                PA_ST = torch.exp(self.QAST_MC.detach() / (Alpha.view(-1,1) + 1e-10) - Z)
                PA_ST = (PA_ST + 1e-10) / ((PA_ST + 1e-10).sum(1, keepdim=True))                
                      
                log_PA_ST = torch.log(PA_ST + 1e-10) 
                HA_ST = -(PA_ST * log_PA_ST).sum(1)
                # HA_ST = -(PA_ST * log_PA_ST).sum(2)
                # log_PA_ST_mean = (PS_s.unsqueeze(2) * log_PA_ST[T[0],:,:].unsqueeze(0)).sum(1).detach()
                HA_ST_mean = (PS_s * HA_ST.view(1,-1)).sum(1, keepdim=True)

                # confidence = 1-torch.exp(-NAST.sum(2) * 5e-3)
                # ri_con = (confidence[T, :] * PS_s * PA_ST[T, :, A]).sum(1, keepdim=True)
                # nqi_cont = torch.min(nqi1_cont, nqi2_cont)
                # nvi_con = (PnA_nsT * nqi_cont).sum(1, keepdim=True)
                # qi_con_approx = ri_con + self.gamma_I * nvi_con # * (1-d)

            # entropy_bonus = (PA_sT * (log_PA_sT + np.log(PA_sT.shape[-1])).detach()).sum(1, keepdim=True)
            pi_loss = (PA_sT * (log_PA_sT - (qt/(alpha+1e-10) - z)).detach()).sum(1, keepdim=True)
            # log_pi_target_local = qt/(alpha+1e-10) - z
            # pi_target_local = torch.exp(log_pi_target_local)
            # pi_loss = ((PA_sT + pi_target_local).detach() * (log_PA_sT - log_pi_target_local.detach())**2).sum(1, keepdim=True)  
            if CG or learn_alpha:
                log_novelty_ratios = self.RND['tl'].novelty_ratios(s, T[0]).detach()             

            if CG:
                S = PS_s.argmax(1).cpu().numpy()
                HS_s = -(PS_s * torch.log(PS_s + 1e-10)).sum(1, keepdim=True)
                concept_entropy_bottleneck = 1 - HS_s.detach() / np.log(self.n_concepts)
                divergence_per_concept = (PA_sT * (log_PA_sT - log_PA_ST[S,:]).detach()).sum(1, keepdim=True)
                # divergence_per_concept = (PA_sT * (log_PA_sT - log_PA_ST[T[0],S,:]).detach()).sum(1, keepdim=True)
                # divergence_per_concept = (PA_sT * (log_PA_sT - log_PA_ST[S,:]).detach()).sum(1, keepdim=True)
                novelty_factor_0 = 1.0 - 1.0 / (1.0 + torch.exp(-2.0 * (log_novelty_ratios - np.log(10)))).view(-1,1)
                # novelty_factor_1 = 1.0 / (1.0 + torch.exp(-5.0 * (log_novelty_ratios - np.log(10)))).view(-1,1)
                # entropy_bottleneck = (1.0 - HA_ST[T[0],S].view(-1,1)/np.log(self.n_skills+1)).detach().clamp(0.0,1.0)
                # corrected_entropy_bottleneck = entropy_bottleneck**novelty_exponent
                total_bottleneck = novelty_factor_0.detach() * concept_entropy_bottleneck
                pi_loss = (1-total_bottleneck) * pi_loss + total_bottleneck * divergence_per_concept # entropy_bottleneck * 
                                
                # novelty_factor = 1.0 - 1.0 / (1.0 + torch.exp(-2.0 * (log_novelty_ratios - np.log(10)))).view(-1,1)
                # pi_target_concept = torch.einsum('ij,jk->ik', PS_s, PA_ST[T[0],:,:])
                # # pi_target_mixed = (1 - novelty_factor) * ideal_PA_sT.detach() + novelty_factor * pi_target_concept
                # log_pi_target_concept = torch.log(pi_target_concept + 1e-10)                
                # pi_loss = pi_loss + novelty_factor * (PA_sT * (log_PA_sT - log_pi_target_concept).detach()).sum(1, keepdim=True)
            
                # divergence_per_concept = (PA_sT.unsqueeze(1) * (log_PA_sT.unsqueeze(1) - log_PA_ST[T[0],:,:].unsqueeze(0)).detach()).sum(2)
                # # novelty_factor = (1.0 - 1.0 / (1.0 + self.novelty_factor_small * log_novelty_ratios)).view(-1,1)
                # novelty_factor = 1.0 / (1.0 + torch.exp(-5.0 * (log_novelty_ratios - np.log(10)))).view(-1,1)
                # entropy_bottleneck = (1.0 - HA_ST[T[0],:].view(1,-1) * novelty_factor/np.log(self.n_skills+1)).detach().clamp(0.0,1.0)
                # # corrected_entropy_bottleneck = entropy_bottleneck**novelty_exponent
                # pi_loss = pi_loss + 0.1 * (PS_s * entropy_bottleneck * divergence_per_concept).sum(1, keepdim=True)
                
                # informative_factor = (1.0 - HA_ST_mean/np.log(self.n_skills+1)).detach().clamp(0.0,1.0)
                # pi_loss = pi_loss + 0.1 * informative_factor * (PA_sT * (log_PA_sT - log_PA_ST_mean).detach()).sum(1, keepdim=True)

                # pi_loss = pi_loss + 0.1 * (PA_sT.detach() * (log_PA_sT - log_PA_ST_mean.detach())**2).sum(1, keepdim=True)
                # pi_loss = pi_loss + 0.1 * (PS_s.unsqueeze(2).detach() * PA_ST.detach()[T[0],:,:].unsqueeze(0) * (log_PA_sT.unsqueeze(1) - log_PA_ST.detach()[T[0],:,:].unsqueeze(0))**2).sum((1,2)).view(-1,1)       

            if learn_alpha:
                log_pi_target = qt.detach()/(alpha+1e-10) - z
                pi_target = torch.exp(log_pi_target)                  
                H_pi_target = -(pi_target * log_pi_target).sum(1, keepdim=True)
                H_pi_target_mean = H_pi_target.mean()
                scaled_min_entropy = self.min_skill_entropy * (1.0 / (1.0 + self.novelty_factor * log_novelty_ratios)) # (0.5*novelty_factor_0.detach() + 0.5)
                alpha_loss = log_alpha * (H_pi_target - scaled_min_entropy.view(-1,1)).detach() # .clamp(-20,2.0)

                scaled_min_entropy_mean = scaled_min_entropy.mean()
                active_concepts = (self.QAST_MC.sum(-1) > 0.0).float().view(-1).detach()
                Alpha_loss = log_Alpha.view(-1) * (HA_ST.view(-1) - 0.5*(scaled_min_entropy_mean + np.log(self.n_skills + 1))).detach() * active_concepts

                # ax = torch.autograd.grad(alpha_loss.mean(), self.CG_actor.log_alpha, retain_graph=True)   

            # if CG:
            #     qe = torch.min(qe1, qe2).detach()
            #     qi_con = torch.min(qi1_con, qi2_con).detach()
            #     Ji = lambda_ * (PA_sT * qi_con).sum(1, keepdim=True)
            #     musk = torch.ones_like(Ji)
            #     dJi_dtheta = torch.autograd.grad(Ji.mean(), self.CG_actor.actor.parameters(), grad_outputs=musk, create_graph=True, retain_graph=True, allow_unused=True)            

            self.CG_actor.qe1.optimizer.zero_grad()
            (qe1_loss * weights).mean().backward()
            clip_grad_norm_(self.CG_actor.qe1.parameters(), self.clip_value)
            self.CG_actor.qe1.optimizer.step()

            self.CG_actor.qe2.optimizer.zero_grad()
            (qe2_loss * weights).mean().backward()
            clip_grad_norm_(self.CG_actor.qe2.parameters(), self.clip_value)
            self.CG_actor.qe2.optimizer.step()
            
            # self.CG_actor.qi1_exploration.optimizer.zero_grad()
            # qi1_exp_loss.mean().backward()
            # clip_grad_norm_(self.CG_actor.qi1_exploration.parameters(), self.clip_value)
            # self.CG_actor.qi1_exploration.optimizer.step()

            # self.CG_actor.qi2_exploration.optimizer.zero_grad()
            # qi2_exp_loss.mean().backward()
            # clip_grad_norm_(self.CG_actor.qi2_exploration.parameters(), self.clip_value)
            # self.CG_actor.qi2_exploration.optimizer.step()

            # if CG:
            #     self.CG_actor.qi1_consensus.optimizer.zero_grad()
            #     qi1_con_loss.mean().backward()
            #     self.CG_actor.qi1_consensus.optimizer.step()

            #     self.CG_actor.qi2_consensus.optimizer.zero_grad()
            #     qi2_con_loss.mean().backward()
            #     self.CG_actor.qi2_consensus.optimizer.step()

            # self.CG_actor.alpha.optimizer.zero_grad()
            # alpha_loss.mean().backward()
            # self.CG_actor.alpha.optimizer.step()

            if learn_alpha:
                self.CG_actor.alpha_optim.zero_grad()
                # self.CG_actor.alpha.optimizer.zero_grad()
                (alpha_loss * weights).mean().backward()
                # clip_grad_norm_(self.CG_actor.alpha.parameters(), self.clip_value)
                self.CG_actor.alpha_optim.step()
                # self.CG_actor.alpha.optimizer.step()
                self.CG_actor.alpha = self.CG_actor.log_alpha.exp()

                self.CG_actor.Alpha_optim.zero_grad()
                (Alpha_loss * weights).mean().backward()
                self.CG_actor.Alpha_optim.step()
                self.CG_actor.Alpha = self.CG_actor.log_Alpha.exp()

            self.CG_actor.actor.optimizer.zero_grad()
            (pi_loss * weights).mean().backward() # retain_graph=CG)
            clip_grad_norm_(self.CG_actor.actor.parameters(), self.clip_value)
            self.CG_actor.actor.optimizer.step()

            # if self.counter['ql'] % 128/4 == 0:
            #     self.RND['tl'].predictor.optimizer.zero_grad()
            #     ri_exp.mean().backward()
            #     clip_grad_norm_(self.RND['tl'].predictor.parameters(), self.clip_value)
            #     self.RND['tl'].predictor.optimizer.step()

            if CG:
                # PA_sT_new = self.CG_actor.actor(s)[0][np.arange(s.size(0)), T, :]
                # Je = (PA_sT_new * qe).sum(1, keepdim=True)
                # dJe_dtheta = torch.autograd.grad(Je.mean(), self.CG_actor.actor.parameters(), retain_graph=True, allow_unused=True)
                # partial_dJe_deta = torch.zeros(1,).to(device)
                # for i in range(0, len(dJe_dtheta)):
                #     partial_dJe_deta = partial_dJe_deta + (dJe_dtheta[i] * dJi_dtheta[i]).sum()
                # lambda_loss = -partial_dJe_deta

                # lg = torch.autograd.grad(lambda_loss, self.CG_actor.lambda_.parameters(), retain_graph=True, allow_unused=True)

                # self.CG_actor.lambda_.optimizer.zero_grad()
                # lambda_loss.mean().backward()
                # self.CG_actor.lambda_.optimizer.step()  

                self.NAST = NAST.detach().clone() 
                self.QAST = QAST.detach().clone()
                self.PA_ST_tl = PA_ST.detach().clone()    

            # if self.counter['ql'] % self.DQL_epsds_target_update == 0:
            #     self.CG_actor.update_targets(1.0)
            #     self.counter['ql'] = 0      

            self.CG_actor.update_targets(self.lr['tl']['target'])     

            if self.per:
                # update priority weights
                self.memory['tl'].update_priority(indices, errors.cpu().numpy())  

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
            re = torch.FloatTensor(trajectory[:,self.s_dim+1]).to(device).view(-1,1)
            ns = torch.FloatTensor(trajectory[:,self.s_dim+2+self.dims['init_ext']:self.s_dim+2+self.dims['last_ext']]).to(device)
            d = torch.FloatTensor(trajectory[:,2*self.s_dim+2]).to(device).view(-1,1)
            T = int(trajectory[0,2*self.s_dim+3])

            self.RND['tl'].update_obs_rms(s, T)
            ri_exp = self.RND['tl'](ns, T).sum(1, keepdim=True)
            self.r_max = max(self.r_max, ri_exp.max().item())

            # mask = torch.rand(len(ri_exp)).to(device)
            # mask = (mask < self.RND_update_proportion).type(torch.FloatTensor).to(device)
            # intrinsic_loss = (ri_exp * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))

            rffs_int = torch.FloatTensor([self.RND['tl'].rff_int.update(rew) for rew in ri_exp.detach().squeeze().tolist()]).to(device)
            self.RND['tl'].rff_rms_int.update(rffs_int, T)
            ri_exp_normalized = ri_exp.detach() / self.RND['tl'].rff_rms_int.var.sqrt()

            mask = torch.rand(len(ri_exp)).to(device)
            mask = (mask < self.RND_update_proportion).type(torch.FloatTensor).to(device)
            intrinsic_loss = (ri_exp * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))

            self.RND['tl'].predictor.optimizer.zero_grad()
            intrinsic_loss.mean().backward()
            clip_grad_norm_(self.RND['tl'].predictor.parameters(), self.clip_value)
            self.RND['tl'].predictor.optimizer.step()

            with torch.no_grad():
                pi_end = self.CG_actor.actor(ns[-1,:].view(1,-1))[0].squeeze(0)[T, :]
                pi_old, log_pi_old = self.CG_actor.actor(s)
                pi_old, log_pi_old = pi_old.detach()[:, T, :], log_pi_old.detach()[:, T, :]
                PA_T = pi_old.mean(0, keepdim=True)
                HA_sT = -(pi_old * log_pi_old).sum(1, keepdim=True)
                HA_sT_mean = HA_sT.mean()            
                        
                qi1_end = self.CG_actor.qi1_exploration(ns[-1,:].view(1,-1)).squeeze(0)[T, :]
                qi2_end = self.CG_actor.qi2_exploration(ns[-1,:].view(1,-1)).squeeze(0)[T, :]
                qi_end = torch.min(qi1_end, qi2_end)
                vi_end = (pi_end * qi_end).sum()

                qi1_exp = self.CG_actor.qi1_exploration(s)[:,T,:]
                qi2_exp = self.CG_actor.qi2_exploration(s)[:,T,:]
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
                qi1_exp_A = self.CG_actor.qi1_exploration(s)[np.arange(N),T,A].view(-1,1)
                qi2_exp_A = self.CG_actor.qi2_exploration(s)[np.arange(N),T,A].view(-1,1)
                qi1_exp_loss = (qi1_exp_A - return_i.detach())**2
                qi2_exp_loss = (qi2_exp_A - return_i.detach())**2

                self.CG_actor.qi1_exploration.optimizer.zero_grad()
                qi1_exp_loss.mean().backward()
                clip_grad_norm_(self.CG_actor.qi1_exploration.parameters(), self.clip_value)
                self.CG_actor.qi1_exploration.optimizer.step()

                self.CG_actor.qi2_exploration.optimizer.zero_grad()
                qi2_exp_loss.mean().backward()
                clip_grad_norm_(self.CG_actor.qi2_exploration.parameters(), self.clip_value)
                self.CG_actor.qi2_exploration.optimizer.step()
            
               
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
                # q1_E, q1_I = self.critic1['sl'](s_batch, a_batch)
                # q1_E, q1_I = q1_E[np.arange(batch_size), T_batch].view(-1,1), q1_I[np.arange(batch_size), T_batch].view(-1,1)
                # q2_E, q2_I = self.critic2['sl'](s_batch, a_batch)
                # q2_E, q2_I = q2_E[np.arange(batch_size), T_batch].view(-1,1), q2_I[np.arange(batch_size), T_batch].view(-1,1)
                # next_v_E, next_v_I = self.v_target['sl'](ns_batch)
                # next_v_E, next_v_I = next_v_E[np.arange(batch_size), T_batch].view(-1,1), next_v_I[np.arange(batch_size), T_batch].view(-1,1)
                
                # RND_error = self.RND['sl'](ns_batch.detach(), T_batch)
                
                # s_approx_batch, mu_latent, log_sigma_latent = self.density(s_batch, T_batch)
                # density_loss = self.density.loss_func(s_batch, s_approx_batch, mu_latent, log_sigma_latent)

                q_approx_E = r_batch + self.gamma_E * next_v_E * (1-d_batch) # + 0.5*RND_error.detach()
                # q_approx_I = angle_loss + self.gamma_I * next_v_I * (1-d_batch)
                # q_approx_I = RND_error.detach() + self.gamma_I * next_v_I * (1-d_batch)
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

                # self.RND['sl'].optimizer.zero_grad()
                # RND_error.mean().backward()
                # clip_grad_norm_(self.RND['sl'].predictor.parameters(), self.clip_value)
                # self.RND['sl'].optimizer.step()

                # self.density.optimizer.zero_grad()
                # density_loss.mean().backward()
                # clip_grad_norm_(self.density.parameters(), self.clip_value)
                # self.density.optimizer.step()                

            # Optimize v network
            a_batch_A, log_pa_sApT_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch_prop.detach())
            if not self.joint:
                A_batch = T_batch
            else:
                A_batch = self.SAC.sample_skills(s_batch, T_batch)
            a_batch_off = a_batch_A[np.arange(batch_size), A_batch, :]
            log_pa_sT = log_pa_sApT_A[np.arange(batch_size), A_batch].view(-1,1)
            
            q1_off_E = self.critic1['sl'](s_batch.detach(), a_batch_off)
            q2_off_E = self.critic2['sl'](s_batch.detach(), a_batch_off)
            # q1_off_E, q1_off_I = self.critic1['sl'](s_batch.detach(), a_batch)
            # q2_off_E, q2_off_I = self.critic2['sl'](s_batch.detach(), a_batch)
            q_off_E = torch.min(torch.stack([q1_off_E, q2_off_E]), 0)[0][np.arange(batch_size), T_batch].view(-1,1)
            # q_off_I = torch.min(torch.stack([q1_off_I, q2_off_I]), 0)[0]
            
            v_approx_E = q_off_E - self.alpha['sl'] * log_pa_sT
            # v_approx_I = q_off_I[np.arange(batch_size), T_batch].view(-1,1) 

            if not only_metrics:
                v_E = self.v['sl'](s_batch)[np.arange(batch_size), T_batch].view(-1,1)
                # v_E, v_I = self.v['sl'](s_batch)
                # v_E, v_I = v_E[np.arange(batch_size), T_batch].view(-1,1), v_I[np.arange(batch_size), T_batch].view(-1,1) 
            
            task_mask = torch.zeros(batch_size, self.n_skills).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            Ha_sT = -(log_pa_sT * task_mask_distribution).sum(0)
            if self.entropy_annealing: alpha_gradient = Ha_sT.detach() - self.threshold_entropy_alpha['sl']

            if not only_metrics:
                v_loss = self.v['sl'].loss_func(v_E.view(-1,1), v_approx_E.view(-1,1).detach()) # + self.v['sl'].loss_func(v_I, v_approx_I.detach())
                self.v['sl'].optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v['sl'].parameters(), self.clip_value)
                self.v['sl'].optimizer.step()
                updateNet(self.v_target['sl'], self.v['sl'], self.lr['sl']['v_target'])
                
                # Optimize skill network
                self.actor.optimizer.zero_grad()
                if not self.surgery:
                    pi_loss = -(v_approx_E).mean()# /self.RND_factor + v_approx_I).mean()                    
                    # pi_loss.backward(retain_graph=True)
                    pi_loss.backward()                    
                else:
                    pi_gradients = []
                    v_per_task = v_approx_E * task_mask_distribution
                    for task in np.unique(T_batch):
                        grad = torch.autograd.grad(-v_per_task[:,task].sum(), self.actor.parameters(), retain_graph=True) # , allow_unused=True
                        pi_gradients.append(grad)
                    for i in range(0, len(pi_gradients)):
                        grad_i = pi_gradients[i]
                        for j in range(0, len(pi_gradients)):
                            if i != j:
                                grad_j = pi_gradients[j]
                                similarities = [(grad_i[k] * grad_j[k]).sum() for k in range(0, len(grad_i))]
                                similarity = sum(similarities)
                                if similarity < 0.0:
                                    norms_j = [(grad_j[k] * grad_j[k]).sum() for k in range(0, len(grad_j))]
                                    norm_j = sum(norms_j)
                                    grad_i = [grad_i[k] - similarity * grad_j[k] / norm_j  for k in range(0, len(grad_i))]
                        for k, param in enumerate(self.actor.parameters()):
                            param.backward(grad_i[k])                
                
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()                    

                if self.entropy_annealing:
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
   
    def learn_concepts(self, meta=False):
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

            
            PS_s, log_PS_s, PnS_ns, log_PnS_ns, _, _, _, _ = self.classifier(s, A_one_hot, ns, explore=True)
            
            # NAST_new = ((PS_s.unsqueeze(2) * A_off_one_hot.unsqueeze(1)).unsqueeze(1) * T_one_hot.unsqueeze(2).unsqueeze(3)).sum(0)
            NAST_new = ((PS_s.unsqueeze(2) * PA_sT.unsqueeze(1)).unsqueeze(1) * T_one_hot.unsqueeze(2).unsqueeze(3)).sum(0)
            NnSAST_new = (((PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)).unsqueeze(1) * T_one_hot.unsqueeze(2).unsqueeze(3)).unsqueeze(4) * PnS_ns.unsqueeze(1).unsqueeze(2).unsqueeze(3)).sum(0)
            df1 = 0.01 * (self.NAST_cl.sum(2, keepdim=True) >= 1.0).float()
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
            HS_T_bound = (PS_s * np.log(self.n_concepts)).sum(1).mean()
            ISs_T = HS_T - HS_s
            
            IS_factor = (1/((self.n_skills+1) * PA_T[T, A])).view(-1,1).detach()
                        
            HA_T_data = -(PA_T_data * torch.log(PA_T_data+1e-10)).sum(1).mean()
            # HA_T = -(torch.log(PA_T_policy+1e-10)[T, A_off].view(-1,1) * PS_s).sum(1).mean()
            # HA_T_data2 = -(torch.log(PA_T_data+1e-10)[T, A_off].view(-1,1) * PS_s).sum(1).mean()
            # HA_ST = -(log_PA_ST[T,:,A_off] * PS_s).sum(1).mean()
            HA_T = -(PA_T_policy * torch.log(PA_T_policy+1e-10)).sum(1).mean()
            HA_ST = -((log_PA_ST[T,:,:] * PA_sT.unsqueeze(1)).sum(2) * PS_s).sum(1).mean()            
            ISA_T = HA_T.detach() - HA_ST
            
            HnS_TdoA_wIS = -(PS_s * (PnS_ns * log_PnS_TdoA.detach()[T, A, :]).sum(1, keepdim=True)).sum(1).mean()
            HnS_TdoA = -(PnS_ns * log_PnS_TdoA[T, A, :]* IS_factor.view(-1,1)).sum(1).mean() #  * IS_factor.view(-1,1)
            HnS_STdoA = -(PS_s * (PnS_ns.unsqueeze(1) * log_PnS_STdoA[T, :, A, :] * IS_factor.view(-1,1,1)).sum(2)).sum(1).mean() #  * IS_factor.view(-1,1,1)
            InSS_TdoA = HnS_TdoA - HnS_STdoA
            
            beta1 = 1.0e-1
            beta2 = 0.25e-1
            alpha2 = 1.0*beta1 / (1-beta2)

            classification_loss = -torch.logsumexp(log_PA_ST[T, :, A_off] + log_PS_s, dim=1).mean()
            model_loss = -torch.logsumexp(torch.log((PnS_STdoA[T, :, A, :] * PnS_ns.unsqueeze(1)).sum(2)+1e-10) + log_PS_s, dim=1).mean()

            classifier_loss = (beta1 + alpha2*beta2) * ISs_T - ISA_T - alpha2 * InSS_TdoA #+ self.lambda_H * HS_s 
            classifier_loss_norm = classifier_loss

            self.classifier.optimizer.zero_grad()
            classifier_loss_norm.backward()
            clip_grad_norm_(self.classifier.parameters(), 1000*self.clip_value)
            self.classifier.optimizer.step()

            self.NAST_cl = NAST.detach().clone()
            self.NnSdoAST_cl = NnSAST.detach().clone()
            X = PA_ST.sum(2)
            p0 = PA_ST[0,:,:]
            p1 = PA_ST[1,:,:]
            p2 = PA_ST[2,:,:]
            p3 = PA_ST[3,:,:]
            
            # log_lambda_H = np.log(self.lambda_H)
            # log_lambda_H += 3e-2*(HS_s.detach().item() - self.lim_HSs)
            # self.lambda_H = np.min([np.max([np.exp(log_lambda_H)+1e-10, 1e-3]), 1.0+(beta1 + alpha2*beta2)]) 
            # self.lim_HSs = np.max([self.lim_HSs * 0.99993, np.log(1.5)])
            
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
        self.params['init_threshold_entropy_alpha'] = self.threshold_entropy_alpha['sl']
        self.params['init_threshold_entropy_alpha_cl'] = self.threshold_entropy_alpha['cl']
        self.params['init_epsilon'] = self.epsilon
        self.params['beta'] = self.beta
        self.params['eta'] = self.eta
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

            if learning_type in ['sl', 'ql']:
                torch.save(self.critic1[learning_type].state_dict(), specific_path+'_critic1_'+learning_type+'.pt')
                if self.stoA_learning_type == 'DQL':
                    torch.save(self.critic2[learning_type].state_dict(), specific_path+'_critic2_'+learning_type+'.pt')                
                    torch.save(self.RND[learning_type].state_dict(), specific_path+'_RDN_'+learning_type+'.pt')
                if self.joint: 
                    torch.save(self.SAC.state_dict(), specific_path+'_SAC_net_'+learning_type+'.pt')
            
            if learning_type == 'sl':
                torch.save(self.v[learning_type].state_dict(), specific_path+'_v_'+learning_type+'.pt')
                torch.save(self.v_target[learning_type].state_dict(), specific_path+'_v_target_'+learning_type+'.pt')
                torch.save(self.actor.state_dict(), specific_path+'_actor_'+learning_type+'.pt')

        if learning_type == 'tl':
            torch.save(self.CG_actor.state_dict(), specific_path+'_transfer_actor_'+learning_type+'.pt')
            torch.save(self.SAC.state_dict(), specific_path+'_transfer_actor_2_'+learning_type+'.pt')
            torch.save(self.RND[learning_type].state_dict(), specific_path+'_RDN_'+learning_type+'.pt')
            pickle.dump(self.NAST,open(specific_path+'_NAST_'+learning_type+'.p','wb'))
            pickle.dump(self.QAST,open(specific_path+'_QAST_'+learning_type+'.p','wb'))
            pickle.dump(self.NAST_MC,open(specific_path+'_NAST_MC_'+learning_type+'.p','wb'))
            pickle.dump(self.QAST_MC,open(specific_path+'_QAST_MC_'+learning_type+'.p','wb'))
            pickle.dump(self.PA_ST_tl,open(specific_path+'_PA_ST_'+learning_type+'.p','wb'))
            # torch.save(self.credibility.state_dict(), specific_path+'_credibility_'+learning_type+'.pt')

        elif learning_type == 'cl':
            suffix = '_we' if self.classification_with_entropies else '_woe'
            torch.save(self.classifier.state_dict(), specific_path+'_classifier' + suffix + '.pt')
            # pickle.dump(self.PS_T,open(specific_path+'_PS_T_'+learning_type+'.p','wb'))
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
                try:
                    self.RND[learning_type].load_state_dict(torch.load(specific_path+'_RDN_'+learning_type+'.pt'))   
                    self.RND[learning_type].train()
                except:
                    pass     

                self.critic1[learning_type].train()
                if self.stoA_learning_type == 'DQL':
                    self.critic2[learning_type].train()                 

                if self.joint: 
                    self.SAC.load_state_dict(torch.load(specific_path+'_SAC_net_'+learning_type+'.pt')) 
                    self.SAC.train()            

        if learning_type == 'tl':
            self.CG_actor.load_state_dict(torch.load(specific_path+'_transfer_actor_'+learning_type+'.pt'))
            self.SAC.load_state_dict(torch.load(specific_path+'_transfer_actor_2_'+learning_type+'.pt'))
            try:
                self.RND[learning_type].load_state_dict(torch.load(specific_path+'_RDN_'+learning_type+'.pt'))   
                self.RND[learning_type].train()
            except:
                pass

        if learning_type == 'cl':
            suffix = '_we' if self.classification_with_entropies else '_woe'
            self.classifier.load_state_dict(torch.load(specific_path+'_classifier' + suffix + '.pt'))
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
            PS_s = self.classifier.sample_concepts(s_cuda, explore=False)[1]
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
        self.SAC = discrete_AC_Net(self.n_skills, self.dims['last_ext']-self.dims['init_ext'], self.n_tasks['ql'], lr=3e-3).to(device)

        eta = self.params['init_eta']
        self.eta['ql'] = (eta['ql'] * torch.ones(self.n_tasks['ql']).float().to(device) if is_float(eta['ql']) else 
                                            (eta['ql'].float().to(device) if is_tensor(eta['ql']) else torch.from_numpy(eta['ql']).float().to(device)))       
    
    def reset_memory(self):
        self.memory['sl'].forget()
        self.memory['ql'].forget()