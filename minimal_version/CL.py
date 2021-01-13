import gym
import torch
import numpy as np
import random

from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from gym.utils import seeding

from utils import scale_action, set_seed
from agent import Agent

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
        set_seed(self.seed, device)
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
            'ql': False,  # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
            'tl': False,
        } 
        self.check_multitask(n_tasks)
        self.agent = Agent(self.s_dim, self.a_dim, n_tasks, agent_params, seed=self.seed, joint=self.params['joint_learning'])               

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
        # self.envs[self.learning_type][self.task].close()        
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
                skill = self.agent.decide(state, task, self.learning_type, explore=explore) # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
            else:
                skill = self.agent.decide(state, task, self.learning_type, explore=explore, rng=self.np_random)   
            s_cuda = torch.FloatTensor(state[self.agent.dims['init_ext']:self.agent.dims['last_ext']]).to(device).view(1,-1)
            with torch.no_grad():
                concept = self.agent.classifier.classifier(s_cuda)[0].argmax().item()
            
        if self.env_names[self.learning_type][self.task] == 'AntCrossMaze-v3':
            self.envs[self.learning_type][self.task]._update_led_visualization(concept, skill)
        
        for env_step in itertools.count(0):
            action = self.agent.act(state, skill, explore=explore if (self.learning_type == 'sl' or self.joint and self.learning_type=='ql') else False, learning_type=self.learning_type)
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
        
        # if self.learning_type == 'tl' and self.active_RND:
        #     ns_cuda = torch.FloatTensor(final_state[self.agent.dims['init_ext']:self.agent.dims['last_ext']]).to(device).view(1,-1)
        #     intrinsic_reward = self.agent.RND['tl'].intrinsic_reward(ns_cuda, self.task)
        #     runnning_return = torch.FloatTensor([self.agent.RND['tl'].discounted_reward.update(intrinsic_reward)]).to(device)
        #     self.agent.RND['tl'].update_q_rms(runnning_return, self.task)
               
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
                if (self.learning_type == 'ql' and (not self.joint or self.agent.stoA_learning_type == 'SAC')) or (self.learning_type == 'tl' and not self.agent.per): self.agent.memorize(event.copy(), self.learning_type)

        if self.learning_type == 'tl' and remember and self.agent.per:  
            with torch.no_grad():              
                s, A, re, ns, d, ri = to_batch(
                    initial_state, skill, total_reward, final_state, masked_done, 0.0,
                    device)
                s, ns = s[:,self.agent.dims['init_ext']:self.agent.dims['last_ext']], ns[:,self.agent.dims['init_ext']:self.agent.dims['last_ext']]
            
                curr_q1 = self.agent.CG_actor.qe1(s)[:,task,skill]
                curr_q2 = self.agent.CG_actor.qe2(s)[:,task,skill]
                next_q1 = self.agent.CG_actor.qe1_target(ns)[:,task,:]
                next_q2 = self.agent.CG_actor.qe2_target(ns)[:,task,:]
                target_q = re
                if not int(masked_done):
                    next_q = torch.min(next_q1, next_q2)
                    next_pi, log_next_pi = self.agent.CG_actor.actor(ns)
                    next_pi, log_next_pi = next_pi[:,task,:], log_next_pi[:,task,:]
                    next_v = (next_pi * (next_q - self.agent.CG_actor.alpha * log_next_pi)).sum(1, keepdim=True)
                    # next_v = (next_pi * (next_q - self.agent.CG_actor.alpha(ns)[0].item() * log_next_pi)).sum(1, keepdim=True)
                    target_q += self.agent.gamma_E * next_v
                error = max(torch.abs(curr_q1 - target_q).item(), torch.abs(curr_q2 - target_q).item())
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.agent.memory['tl'].append(
                    initial_state, skill, total_reward, final_state, masked_done, 0.0,
                    error, episode_done=end_step)

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
            if self.keep_track: trajectory = []
            if self.keep_track and self.active_MC: trajectory_MC = []
            
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
                        elif not self.active_RND:
                            trajectories.append(np.array(trajectory.copy()))
                            trajectory = []             
                step_counter += env_steps

                if (epsd_step + 1) % (512) == 0 and self.agent.active_intrinsic_learning: 
                    self.agent.intrinsic_learning(trajectory)
                    trajectory = []

                if step_counter >= self.steps['tr'][self.learning_type] * self.steps['env'][self.learning_type]: 
                    if self.keep_track:                    
                        if len(trajectory) >= 1:
                            if self.active_RND and self.agent.active_intrinsic_learning:
                                self.agent.intrinsic_learning(trajectory)                            
                            elif not self.active_RND:
                                trajectories.append(np.array(trajectory.copy()))
                            trajectory = []
                            if self.active_MC:
                                self.agent.MC_learning(trajectory_MC)
                                trajectory_MC = []
                    break
                    
                if self.multitask_envs[self.learning_type] and ((step_counter+1) % self.steps['MT']) == 0: 
                    self.MT_task = (self.MT_task + np.random.randint(self.n_MT_tasks-1) + 1) % self.n_MT_tasks
                    if self.keep_track and not self.active_RND:
                        if len(trajectory) >= 1:
                            trajectories.append(np.array(trajectory.copy()))
                            trajectory = [] 
                # if epsd_step >= self.steps['tr'][self.learning_type]: break
            
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
            # if self.joint: self.agent.reset_memory()
    
    def train_agent_concepts(self, losses=[], entropies=[], entropies_2=[], storing_path='', iter_0=0, min_lr=1e-4, max_lr=3e-4, T=50000, max_tau=5.0, min_tau=1.3, last_steps=50000, max_max_tau=10.0):
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()

        suffix = '_we' if self.agent.classification_with_entropies else '_woe'

        for grad_step in range(0, self.steps['tr'][self.learning_type]):
            classifier_loss, HS_T,  HS_s, ISs_T, ILBO_AS_T_term1, HA_T, ILBO_AS_T, ILBO_nSS_TdoA_term1, HnS_TdoA, ILBO_nSS_TdoA, cl, ml = self.agent.learn_concepts()
            losses.append([classifier_loss, cl, ml])
            entropies.append([HS_T,  HS_s, ISs_T, ILBO_AS_T_term1, HA_T, ILBO_AS_T, ILBO_nSS_TdoA_term1, HnS_TdoA, ILBO_nSS_TdoA])

            stdscr.addstr(0, 0, "Iteration: {}".format(grad_step))
            stdscr.addstr(1, 0, "Classifier Loss: {}".format(np.round(classifier_loss, 4)))
            stdscr.addstr(2, 0, "Entropy H(S|T): {}".format(np.round(HS_T, 4)))
            stdscr.addstr(3, 0, "Entropy H(S|s): {}".format(np.round(HS_s,4)))
            stdscr.addstr(4, 0, "MutualInfo I(S:s|T): {}".format(np.round(ISs_T,4)))
            # stdscr.addstr(5, 0, "Expected log P(A|S,T) : {}".format(np.round(ILBO_AS_T_term1,4)))
            stdscr.addstr(5, 0, "Entropy H(A|S,T): {}".format(np.round(ILBO_AS_T_term1,4)))
            stdscr.addstr(6, 0, "Entropy H(A|T): {}".format(np.round(HA_T,4)))
            stdscr.addstr(7, 0, "MutualInfo I(A:S|T): {}".format(np.round(ILBO_AS_T,4)))
            # stdscr.addstr(7, 0, "ILBO I(A:S|T): {}".format(np.round(ILBO_AS_T,4)))
            # stdscr.addstr(8, 0, "Expected log P(S'|S,T,do(A)) : {}".format(np.round(ILBO_nSS_TdoA_term1,4)))
            stdscr.addstr(8, 0, "Entropy H(S'|S,T,do(A)) : {}".format(np.round(ILBO_nSS_TdoA_term1,4)))
            stdscr.addstr(9, 0, "Entropy H(S'|T,do(A)): {}".format(np.round(HnS_TdoA,4)))
            # stdscr.addstr(10, 0, "ILBO I(S':S|T,do(A)): {}".format(np.round(ILBO_nSS_TdoA,4)))
            stdscr.addstr(10, 0, "MutualInfo I(S':S|T,do(A)): {}".format(np.round(ILBO_nSS_TdoA,4)))
            stdscr.addstr(11, 0, "Policy model loss: {}".format(np.round(cl, 4)))
            stdscr.addstr(12, 0, "Transition model loss: {}".format(np.round(ml, 4)))
            stdscr.refresh()

            if (grad_step + 1) % 5000 == 0:
                self.save(storing_path, storing_path+ '/' + str(iter_0+grad_step+1))
                np.savetxt(storing_path + '/concept_training_losses' + suffix + '.txt', np.array(losses))
                np.savetxt(storing_path + '/concept_training_entropies' + suffix + '.txt', np.array(entropies)) 
                # np.savetxt(storing_path + '/concept_training_entropies_2' + suffix + '.txt', np.array(entropies_2)) 

            # if grad_step <= T:
            #     self.agent.classifier.optimizer.param_groups[0]['lr'] = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos(np.pi * ((grad_step)/T)))
            #     if grad_step <= T-last_steps:
            #         self.agent.classifier.tau = min_tau + (max_tau-min_tau)*np.absolute(1  - grad_step/(T//2))
            #     else:
            #         self.agent.classifier.tau = np.min([max_max_tau, self.agent.classifier.tau*1.00002])
            # self.agent.classifier.optimizer.param_groups[0]['lr'] = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos(np.pi * ((grad_step%T)/T)))

        curses.echo()
        curses.nocbreak()
        curses.endwin()
        
        # self.save(storing_path, storing_path+ '/' + str(iter_0+self.steps['tr'][self.learning_type]))
        # np.savetxt(storing_path + '/concept_training_losses' + suffix + '.txt', np.array(losses))
        # np.savetxt(storing_path + '/concept_training_entropies' + suffix + '.txt', np.array(entropies))            

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
        # if self.env_names[self.learning_type][self.task] == 'AntCrossMaze-v3': max_step *= 3

        for epsd in range(0, eval_epsds):
            step_counter = 0

            if self.store_video: video = VideoWriter(specific_path + '_' + str(self.task) + '_' + str(epsd) + '_' + self.learning_type + '.avi', fourcc, float(FPS), (width, height))

            change_env = False if epsd == 0 or given_task else True
            self.reset(change_env=change_env)           
            epsd_reward = 0.0
            previous_skill = self.task

            for eval_step in itertools.count(0):            
                reward, done, event, env_steps, previous_skill = self.interaction(remember=False, explore=explore, learn=False, lim=self.steps['tr'][self.learning_type]-step_counter, previous_skill=previous_skill)
                # reward = 0.0
                # done = False
                # env_steps = 5
                # event = np.empty(2*self.s_dim+4)
                # previous_skill = 0
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

        

