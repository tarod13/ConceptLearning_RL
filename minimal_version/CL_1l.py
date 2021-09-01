import random
import pickle
import itertools
from sys import stdout

import numpy as np
import torch

import gym
from gym.utils import seeding

from utils import scale_action, set_seed
from agent_1l import Agent

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

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
    def __init__(self, params, agent_params={}):
        
        self.params = params
        default_params = {
                            'seed': 1000,
                            'env_names': [],
                            'env_steps': 1, 
                            'grad_steps': 1, 
                            'init_steps': 10000,
                            'max_episode_steps': 1000,
                            'tr_steps': 1000,
                            'tr_epsd': 4000,
                            'eval_epsd': 10,
                            'eval_epsd_interval': 20,
                            'eval_steps': 1000,
                            'batch_size': 256, 
                            'render': True, 
                            'reset_when_done': True, 
                            'store_video': False,
                            'storing_path': '',
                            'MT_steps': 200,
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
        self.env_names = self.params['env_names']
        self.n_tasks = len(self.env_names)
        self.steps = {
                        'env': self.params['env_steps'],
                        'grad': self.params['grad_steps'],
                        'init': self.params['init_steps'],
                        'tr': self.params['tr_steps'],
                        'MT': self.params['MT_steps'],
                        'eval': self.params['eval_steps'],
                    }
        self.epsds = {
            'tr': self.params['tr_epsd'],
            'eval': self.params['eval_epsd'],
            'eval_interval': self.params['eval_epsd_interval']
        }
        
        self.batch_size = self.params['batch_size']
        self.render = self.params['render']
        self.store_video = self.params['store_video']
        self.reset_when_done = self.params['reset_when_done']
        self._max_episode_steps = self.params['max_episode_steps']
        
        self.envs = {}
        self.active_RND = self.params['active_RND']
        self.active_MC = self.params['active_MC']
        self.masked_done = self.params['masked_done']

        self.set_envs()

        self.s_dim = self.envs[0].observation_space.shape[0]
        self.a_dim = self.envs[0].action_space.shape[0]        
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = self.s_dim*2 + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 1
        self.epsd_counter = 0
        self.task = 0
        self.MT_task = 0

        self.min_action = self.envs[0].action_space.low[0]
        self.max_action = self.envs[0].action_space.high[0]

        self.n_MT_tasks = self.n_tasks
        self.multitask_envs = False
        self.check_multitask()
        n_tasks = self.n_MT_tasks if self.multitask_envs else self.n_tasks
        self.agent = Agent(self.s_dim, self.a_dim, n_tasks, agent_params, seed=self.seed)               

    def check_multitask(self):
        if self.n_tasks == 1:
            try:
                n = self.envs[0]._n_tasks
                self.multitask_envs = True
                self.n_MT_tasks = n
            except:
                pass 

    def set_envs(self):
        self.envs = []        
        for i in range(0, self.n_tasks):                    
            self.envs.append(gym.make(self.env_names[i]).unwrapped)
            print("Created env "+self.env_names[i])
            self.envs[i].reset()
            self.envs[i].seed(self.seed)        
            self.envs[i]._max_episode_steps = self._max_episode_steps
            self.envs[i].rgb_rendering_tracking = True
    
    def reset(self, change_env=False):
        if change_env: self.task = (self.task+1) % self.n_tasks
        self.envs[self.task].reset()        
    
    def get_obs(self):
        state = self.envs[self.task]._get_obs().copy()
        return state
     
    def initialization(self, init_steps=0):         
        self.reset()
        skill = 0
        if init_steps == 0: init_steps = self.steps['init']
        for init_step in range(0, init_steps * self.n_tasks):
            if self.multitask_envs:
                if (init_step % self.steps['env']) == 0 and np.random.rand()>0.95:
                    skill = np.random.randint(self.agent.n_skills)
            else:
                skill = self.task
            done = self.interaction_init(skill)
            limit_reached = (init_step+1) % init_steps == 0
            if done or limit_reached: self.reset(change_env=limit_reached)
            if self.render: self.envs[self.task].render()                        
        print("Finished initialization...")

    def interaction_init(self, skill):  
        event = np.empty(self.t_dim)
        state = self.get_obs()
        action = 2.0*np.random.rand(self.a_dim)-1.0        
        next_state, reward, done, info = self.envs[self.task].step(action)  
        done = done and self.reset_when_done
        if self.multitask_envs:
            skill_reward = info['reward_'+str(skill)]
            reward += skill_reward

        event[:self.s_dim] = state
        event[self.s_dim:self.sa_dim] = action
        event[self.sa_dim] = reward
        event[self.sa_dim+1:self.sars_dim] = next_state
        event[self.sars_dim] = float(done)
        event[self.sarsd_dim] = skill
        
        self.agent.memorize(event)   
        return done

    def interaction(self, remember=True, explore=True, learn=True, previous_skill = 0):  
        event = np.empty(self.t_dim)
        initial_state = self.get_obs()
        state = initial_state.copy()
        total_reward = 0.0
        done = end_step = False
        max_env_step = self.steps['env']
        
        task = self.MT_task if self.multitask_envs else self.task

        try:
            self.envs[self.task]._update_quaternion()
        except:
            pass
        
        if self.multitask_envs:
            if np.random.rand() > 0.95:
                skill = np.random.randint(self.agent.n_skills)
            else:
                skill = previous_skill
        else:
            skill = task
        
        for env_step in itertools.count(0):
            action = self.agent.act(state, skill, explore=explore)
            scaled_action = scale_action(action, self.min_action, self.max_action).reshape(-1)
            next_state, reward, done, info = self.envs[self.task].step(scaled_action)
            if self.multitask_envs:
                skill_reward = info['reward_'+str(skill)]
                reward += skill_reward
            
            end_step = end_step or (done and self.reset_when_done)
            total_reward += reward 

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.sars_dim] = next_state
            event[self.sars_dim] = float(done)
            event[self.sarsd_dim] = skill
        
            if remember: self.agent.memorize(event.copy())
            if env_step < self.steps['env']-1: state = np.copy(next_state)
            if end_step or (env_step+1) >= max_env_step: break 
            if self.render and ((env_step+1)%10) == 0: self.envs[self.task].render()

        if learn:
            for _ in range(0, self.steps['grad']):
                self.agent.learn_skills()
            
        return total_reward, done, event, env_step+1, skill 

    def train_agent(self, initialization=True, storing_path='', rewards=[], metrics=[], iter_0=0):
        if len(storing_path) == 0: storing_path = self.params['storing_path']

        if initialization:
            self.initialization()
            specific_path = storing_path + '/' + str(0)
            self.save(storing_path, specific_path)
        
        init_iter = iter_0
        self.train_agent_skills(storing_path=storing_path, rewards=rewards, metrics=metrics, iter_0=init_iter)

    def train_agent_skills(self, iter_0=0, rewards=[], metrics=[], storing_path=''):        
        if self.render: self.envs[self.task].render()   
        
        lim_epsd = self.epsds['tr']
        for epsd in range(0, lim_epsd):
            change_env = False if epsd == 0 else True
            self.reset(change_env=change_env)
            iter_ = iter_0 + (epsd+1) // self.epsds['eval_interval']
            step_counter = 0
            previous_skill = self.task
            
            for epsd_step in itertools.count(0):
                learn = epsd != 0 or epsd_step+1 > 3*self.batch_size
                done, _, env_steps, previous_skill = self.interaction(learn=learn, previous_skill=previous_skill)[1:]

                if self.render: self.envs[self.task].render()                
                if done: self.reset(change_env=False)                                 
                step_counter += env_steps

                if step_counter >= self.steps['tr'] * self.steps['env']: break
                    
                if self.multitask_envs and ((step_counter+1) % self.steps['MT']) == 0: 
                    self.MT_task = (self.MT_task + np.random.randint(self.n_MT_tasks-1) + 1) % self.n_MT_tasks
                    
            if (epsd+1) % self.epsds['eval_interval'] == 0:
                st0_random = random.getstate()
                st0 = np.random.get_state() 
                st0_rng = self.np_random.get_state()
                st0_torch = torch.get_rng_state() 
                if device == "cuda": st0_torch_cuda = torch.cuda.get_rng_state()  
                st_envs = []
                for env in self.envs:
                    st_envs.append(env.np_random.get_state())
                                 
                r, _, m, l = self.eval_agent_skills(explore=True, iter_=iter_, store_events=False)
                random.setstate(st0_random)
                np.random.set_state(st0)
                self.np_random.set_state(st0_rng)
                torch.set_rng_state(st0_torch)
                if device == "cuda": torch.cuda.set_rng_state(st0_torch_cuda)
                for i, env in enumerate(self.envs):
                    env.np_random.set_state(st_envs[i])

                metrics.append(m)
                rewards += r
                np.savetxt(storing_path + '/metrics.txt', np.array(metrics))               
                
                specific_path = storing_path + '/' + str(iter_)
                self.save(storing_path, specific_path=specific_path)
                np.savetxt(storing_path + '/mean_rewards.txt', np.array(rewards))


    def eval_agent_skills(self, eval_epsds=0, explore=False, iter_=0, start_render=False, print_space=True, specific_path='video', max_step=0, task=None, store_events=True):
        if task is None:   
            task = self.task
            self.task = 0
            given_task = False
            if self.multitask_envs:
                MT_task = self.MT_task
                self.MT_task = 0
        else:
            self.task = task
            given_task = True
        self.reset()

        if start_render: self.envs[self.task].render()
        if eval_epsds == 0: 
            n_tasks = self.n_MT_tasks if self.multitask_envs else self.n_tasks
            eval_epsds = self.epsds['eval'] * n_tasks
        
        events = []
        rewards = []
        epsd_lengths = []
        min_epsd_reward = 1.0e6
        max_epsd_reward = -1.0e6
        
        Ha_sT = []
        Ha_sT_average = 0.0
        
        if max_step <= 0: max_step = self.steps['eval']

        for epsd in range(0, eval_epsds):
            if self.store_video: video = VideoWriter(
                specific_path + '_' + str(self.task) + '_' + str(epsd) + '.avi', 
                fourcc, float(FPS), (width, height))

            change_env = False if epsd == 0 or given_task else True
            self.reset(change_env=change_env)           
            epsd_reward = 0.0
            previous_skill = self.task

            for eval_step in itertools.count(0):            
                reward, done, event, _, previous_skill = self.interaction(
                    remember=False, explore=explore, learn=False, 
                    previous_skill=previous_skill
                )
                event[self.sa_dim] = reward  
                epsd_reward += reward                                      

                if self.store_video:
                    img = self.envs[self.task].render('rgb_array',1024,768)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif self.render:
                    self.envs[self.task].render()

                if store_events:
                    if self.env_names[self.task] in ['AntGather-v3', 'AntAvoid-v3']:
                        object_positions = np.copy(self.envs[self.task]._object_positions[:,:2].reshape(-1))
                        event = np.concatenate([event, object_positions])
                    events.append(event)

                if done or ((eval_step + 1) >= max_step):
                    epsd_lengths.append(eval_step + 1)
                    break

            metrics = self.agent.estimate_metrics()                                      
            Ha_sT.append(metrics['H(a|s,T)'])
            Ha_sT_average += (Ha_sT[-1] - Ha_sT_average)/(epsd+1)

            rewards.append(epsd_reward)
            min_epsd_reward = np.min([epsd_reward, min_epsd_reward])
            max_epsd_reward = np.max([epsd_reward, max_epsd_reward])
            average_reward = np.array(rewards).mean()           
            
            stdout.write("Iter %i, epsd %i, H(a|s,T): %.4f, min r: %i, max r: %i, mean r: %i, epsd r: %i\r " %
                (iter_, (epsd+1), Ha_sT_average, min_epsd_reward//1, max_epsd_reward//1, average_reward//1, epsd_reward//1))
            stdout.flush()  

            self.MT_task = (self.MT_task + 1) % self.n_MT_tasks

        if print_space: print("")

        if self.store_video: video.release()
        metric_vector = np.array([Ha_sT_average])
        
        if not given_task: 
            self.task = task
            if self.multitask_envs:
                self.MT_task = MT_task
        return rewards, np.array(events), metric_vector, np.array(epsd_lengths)   

    
    def save(self, path, specific_path=''):
        pickle.dump(self.params, open(path+'/params.p','wb'))
        self.agent.save(path, specific_path)
    
    def load(self, path, iter=0, load_memory=True):
        try:
            self.agent.load(path, path + '/' + str(iter), load_memory=load_memory)
        except:
            raise RuntimeError('Could not find agent at '+path + '/' + str(iter))
            

        

