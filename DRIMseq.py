import gym
import torch
import numpy as np
from scipy.special import logsumexp

import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.nn.utils import clip_grad_norm_

from nets import (Memory, v_valueNet, q_valueNet, policyNet, conditionalEncoder, mixtureConceptModel, encoderConceptModel, 
                    rNet, rewardNet, v_parallel_valueNet, q_parallel_valueNet, nextSNet, conceptLossNet, transitionNet, 
                    r_parallelNet, SimPLeNet, ConditionalSimPLeNet, AutoregressivePrior, ConditionalVQVAE_Net, classifierConceptModel)
from tabularQ import Agent as mAgent

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

plt.ion()
# writer = SummaryWriter()
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
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)


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
                            'alpha': 1.0,
                            'init_threshold_entropy_alpha': 0.0,
                            'delta_threshold_entropy_alpha': 1.6e-5,

                            'lr': {
                                    'cl': {
                                            'q': 3e-4,
                                            'v': 3e-4,
                                            'pi': 3e-4,
                                            'alpha': 3e-4,
                                            'v_target': 5e-3
                                        },
                                    'sl': {
                                            'q': 3e-4,
                                            'v': 3e-4
                                        }
                                    },

                            'dim_excluded': {
                                        'init': 2,
                                        'last': 40
                                    },
                            
                            'batch_size': {
                                            'cl': 256,
                                            'sl': 256
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
        self.n_skills = n_tasks['cl']

        self.n_concepts = self.params['n_concepts']
        self.dim_excluded = self.params['dim_excluded']
        self.batch_size = self.params['batch_size']
        self.lr = self.params['lr']
        self.gamma = self.params['gamma']
        self.clip_value = self.params['clip_value']

        # Metric weights
        self.min_threshold_entropy_alpha = -a_dim*1.0
        self.threshold_entropy_alpha = self.params['init_threshold_entropy_alpha']
        self.delta_threshold_entropy_alpha = self.params['delta_threshold_entropy_alpha']
        if isinstance(self.params['alpha'], float):
            self.alpha = self.params['alpha'] * torch.ones(self.n_tasks['cl']).float().to(device)
        elif isinstance(self.params['alpha'], torch.FloatTensor) or isinstance(self.params['alpha'], torch.Tensor):
            self.alpha = self.params['alpha'].float().to(device)
        else:
            self.alpha = torch.from_numpy(self.params['alpha']).float().to(device)
        
        
        # Nets and memory
        self.critic1 = {
                            'cl': q_valueNet(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, n_tasks['cl'], lr=self.lr['cl']['q']).to(device),
                            'sl': q_valueNet(s_dim, a_dim, n_tasks['sl'], lr=self.lr['sl']['q']).to(device)
                        }
        self.critic2 = {
                            'cl': q_valueNet(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, n_tasks['cl'], lr=self.lr['cl']['q']).to(device),
                            'sl': q_valueNet(s_dim, a_dim, n_tasks['sl'], lr=self.lr['sl']['q']).to(device)
                        }
        self.v = {
                            'cl': v_valueNet(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), n_tasks['cl'], lr=self.lr['cl']['v']).to(device),
                            'sl': v_valueNet(s_dim, n_tasks['sl'], lr=self.lr['sl']['v']).to(device)
                        }
        self.v_target = {
                            'cl': v_valueNet(s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), n_tasks['cl'], lr=self.lr['cl']['v']).to(device),
                            'sl': v_valueNet(s_dim, n_tasks['sl'], lr=self.lr['sl']['v']).to(device)
                        }
        self.actor = policyNet(self.n_skills, s_dim-(self.dim_excluded['init']+self.dim_excluded['last']), a_dim, lr=self.lr['cl']['pi']).to(device)

        self.memory = {
                        'cl':  Memory(self.params['memory_capacity'], n_seed=self.seed),
                        'sl':  Memory(self.params['memory_capacity'], n_seed=self.seed)
                    }    
    
    def memorize(self, event, learning_type, init=False):
        if init:
            self.memory[learning_type].store(event[np.newaxis,:])
        else:
            self.memory[learning_type].store(event.tolist())
    
    def learn_skills(self, only_metrics=False):
        batch = self.memory['cl'].sample(self.batch_size['cl'])
        batch = np.array(batch)
        batch_size = batch.shape[0]

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dim_excluded['init']:self.s_dim-self.dim_excluded['last']]).to(device)
            a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
            r_batch = torch.FloatTensor(batch[:,self.sa_dim]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.sa_dim+1+self.dim_excluded['init']:self.sars_dim-self.dim_excluded['last']]).to(device)
            d_batch = torch.FloatTensor(batch[:,self.sars_dim]).view(-1,1).to(device)
            T_batch = batch[:,self.sarsd_dim+1].astype('int')  

            if not only_metrics:
                # Optimize q networks
                q1 = self.critic1['cl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q2 = self.critic2['cl'](s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                next_v = self.v_target['cl'](ns_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q_approx = r_batch + self.gamma * next_v * (1-d_batch)
                
                q1_loss = self.critic1['cl'].loss_func(q1, q_approx.detach())
                self.critic1['cl'].optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1['cl'].parameters(), self.clip_value)
                self.critic1['cl'].optimizer.step()
                
                q2_loss = self.critic2['cl'].loss_func(q2, q_approx.detach())
                self.critic2['cl'].optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2['cl'].parameters(), self.clip_value)
                self.critic2['cl'].optimizer.step()                

            # Optimize v network
            a_batch_A, log_pa_sApT_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch)
            a_batch = a_batch_A[np.arange(batch_size), T_batch, :]
            log_pa_sT = log_pa_sApT_A[:, :, T_batch]
            
            q1_off = self.critic1['cl'](s_batch.detach(), a_batch)
            q2_off = self.critic2['cl'](s_batch.detach(), a_batch)
            q_off = torch.min(torch.stack([q1_off, q2_off]), 0)[0]
            
            v_approx = (q_off - self.alpha.view(1,-1) * log_pa_sT)[np.arange(batch_size), T_batch].view(-1,1) 

            if not only_metrics:
                v = self.v['cl'](s_batch)[np.arange(batch_size), T_batch].view(-1,1)
            
            task_mask = torch.zeros(batch_size, self.n_tasks).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            Ha_sT = -(log_pa_sT * task_mask_distribution).sum(0, keepdim=True)
            alpha_gradient = Ha_sT.detach() - self.threshold_entropy_alpha

            if not only_metrics:
                v_loss = ((v - v_approx.detach())**2).mean()
                self.v['cl'].optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v['cl'].parameters(), self.clip_value)
                self.v['cl'].optimizer.step()
                updateNet(self.v_target['cl'], self.v['cl'], self.lr['cl']['v_target'])

                # Optimize skill network
                pi_loss = (-v_approx).mean()
                self.actor.optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()

                # Optimize dual variable                
                log_alpha = torch.log(self.alpha + 1e-6)
                log_alpha -= self.lr['cl']['alpha'] * alpha_gradient
                self.alpha = torch.exp(log_alpha).clamp(1e-10, 1e+3)

                self.threshold_entropy_alpha = np.max([self.threshold_entropy_alpha - self.delta_threshold_entropy_alpha, self.min_threshold_entropy_alpha])
                    
        else:
            log_pa_sT = torch.zeros(1).to(device)  
            Ha_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(a|sT)': Ha_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics
    
    
    def save(self, common_path, specific_path, learning_type):
        self.params['alpha'] = self.alpha
        self.params['threshold_entropy_alpha'] = self.threshold_entropy_alpha
        
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
        if learning_type == 'cl':
            torch.save(self.actor.state_dict(), specific_path+'_actor.pt')        
                
    
    def load(self, common_path, specific_path, learning_type, load_memory=True, load_upper_memory=True):
        if load_memory: 
            data_batches = pickle.load(open(common_path+'/data_batches_'+learning_type+'.p','rb'))
            pointer = 0
            for i in range(0, data_batches['l']):
                data = pickle.load(open(common_path+'/memory_'+learning_type+str(i+1)+'.p','rb'))
                self.memory[learning_type].data += data
                pointer += len(data)
            self.memory.pointer = pointer % self.memory[learning_type].capacity

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
    def __init__(self, params, agent_params={}):
        
        self.params = params
        default_params = {
                            'seed': 1000,
                            'env_names_sl': ['Hopper-v2'],
                            'env_names_cl': ['Hopper-v2'],
                            'env_names_tl': ['Hopper-v2'],
                            'env_steps': 1, 
                            'grad_steps': 1, 
                            'init_steps': 10000,
                            'max_episode_steps': 1000,
                            'batch_size': 256, 
                            'render': True, 
                            'reset_when_done': True, 
                            'store_video': False                            
                        }

        for key, value in default_params.items():
            if key not in self.params.keys():
                self.params[key] = value

        self.seed = self.params['seed']
        set_seed(self.seed)
        self.env_names = {
                            'sl': self.params['env_names_sl'],
                            'cl': self.params['env_names_cl'],
                            'tl': self.params['env_names_tl']
                        }
        self.n_tasks = {
                            'sl': len(self.env_names['sl']),
                            'cl': len(self.env_names['cl']),
                            'tl': len(self.env_names['tl'])
                        }
       
        self.env_steps = self.params['env_steps']
        self.grad_steps = self.params['grad_steps']
        self.init_steps = self.params['init_steps']
        self.batch_size = self.params['batch_size']
        self.render = self.params['render']
        self.store_video = self.params['store_video']
        self.reset_when_done = self.params['reset_when_done']
        self._max_episode_steps = self.params['max_episode_steps']

        self.envs = {}

        self.set_envs('sl')

        self.s_dim = self.envs['sl'][0].observation_space.shape[0]
        self.a_dim = self.envs['sl'][0].action_space.shape[0]        
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = self.s_dim*2 + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 2
        self.epsd_counter = 0
        self.task = 0

        self.min_action = self.envs['sl'][0].action_space.low[0]
        self.max_action = self.envs['sl'][0].action_space.high[0]

        self.agent = Agent(self.s_dim, self.a_dim, self.n_tasks, agent_params, seed=self.seed) 

    def set_envs(self, learning_type):
        self.envs[learning_type] = []        
        for i in range(0, self.n_tasks[learning_type]):                    
            self.envs[learning_type].append(gym.make(self.env_names[learning_type][i]).unwrapped)
            print("Created env "+self.env_names[learning_type][i])
            self.envs[learning_type][i].reset()
            self.envs[learning_type][i].seed(self.seed)        
            self.envs[learning_type][i]._max_episode_steps = self._max_episode_steps
            self.envs[learning_type][i].rgb_rendering_tracking = True
    
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
            if done or (init_step+1) % (self.init_steps//self.n_tasks) == 0:
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
        
        self.agent.memorize(event)   
        return event

    def interaction(self, learn_upper=True, remember=True, init=False, explore=True, epsd_step=0, learn_lower=True, transfer=False):  
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
        
            assert isinstance(reward, float), "problems in paradise"

            if self.multitask:
                event[self.sarsd_dim] = reward # info['reward_goal']                  
            else:
                event[self.sarsd_dim] = reward    
            
            if remember:
                self.agent.memorize(event)

            if self.hierarchical:
                self.agent.update_upper_level(reward, done, self.task, (epsd_step*self.env_steps+env_step+1)>=self.envs[self.task]._max_episode_steps, 
                    state, action, action_llhood, remember=remember, learn=learn_upper, transfer=transfer)

            if done:                
                break

            if env_step < self.env_steps-1:
                state = np.copy(next_state)
        
        if learn_lower and not init:
            for _ in range(0, self.grad_steps):
                self.agent.learn_lower()

        return event, done

    def train(self):
        pass
    
    def train_agent(self, tr_epsds, epsd_steps, initialization=True, eval_epsd_interval=10, eval_epsds=12, iter_=0, save_progress=True, common_path='', 
        rewards=[], goal_rewards=[], metrics=[], learn_lower=True, transfer=False, model_iter=10000, model_epsd_interval=10):        
        if self.render:
            self.envs[self.task].render()

        if initialization:
            self.initialization(epsd_steps)

            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()

            for i in range(0, 10*model_iter):
                self.agent.learn_transition_model(i, stdscr)

            curses.echo()
            curses.nocbreak()
            curses.endwin()

            specific_path = common_path + '/' + str(0)
            self.save(common_path, specific_path)

        n_done = 0

        for epsd in range(0, tr_epsds):
            self.epsd_counter += 1
            if epsd == 0:
                self.reset(change_env=False)
            else:
                self.reset(change_env=True)
            
            for epsd_step in range(0, epsd_steps):
                if len(self.agent.memory.data) < self.batch_size:
                    done = self.interaction(learn_upper=False, learn_lower=False, epsd_step=epsd_step, transfer=transfer)[1]
                else:
                    done = self.interaction(learn_upper=True, learn_lower=learn_lower, epsd_step=epsd_step, transfer=transfer)[1]

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
            
            if (epsd+1) % model_epsd_interval == 0:
                if self.hierarchical:
                    stdscr = curses.initscr()
                    curses.noecho()
                    curses.cbreak()

                    for i in range(0, model_iter):
                        self.agent.learn_transition_model(i, stdscr)

                    curses.echo()
                    curses.nocbreak()
                    curses.endwin()

                    if save_progress:
                        specific_path = common_path + '/' + str(iter_ + (epsd+1) // eval_epsd_interval)
                        self.save(common_path, specific_path)
                        np.savetxt(common_path + '/mean_rewards.txt', np.array(rewards))
              
        if self.multitask:
            return np.array(rewards).reshape(-1), np.array(goal_rewards).reshape(-1)
        else:      
            return np.array(rewards).reshape(-1)      
    
    def save(self, common_path, specific_path):
        pickle.dump(self.params,open(common_path+'/params.p','wb'))
        self.agent.save(common_path, specific_path)
    
    def load(self, common_path, specific_path, load_memory=True, load_upper_memory=True, transfer=False):
        self.agent.load(common_path, specific_path, load_memory=load_memory, load_upper_memory=load_upper_memory, transfer=transfer)

