import collections
import numpy as np
import torch
import random

ExperienceFirstLevel = collections.namedtuple(
    'ExperienceFirstLevel', field_names=['state', 'action', 'reward', 'done', 'next_state'])
PixelExperienceSecondLevel = collections.namedtuple(
    'PixelExperienceSecondLevel', field_names=['inner_state', 'outer_state', 'action', 'reward', 
                                'done', 'next_inner_state', 'next_outer_state'])
PixelExperienceThirdLevel = collections.namedtuple(
    'PixelExperienceSecondLevel', field_names=['inner_state', 'outer_state', 'action', 'action_distribution', 'reward', 
                                'done', 'next_inner_state', 'next_outer_state'])
PixelExperienceSecondLevelMT = collections.namedtuple(
    'PixelExperienceSecondLevelMT', field_names=['inner_state', 'outer_state', 'action', 'reward', 
                                'done', 'next_inner_state', 'next_outer_state', 'task'])
Task = collections.namedtuple('Task', field_names=['task'])

class ExperienceBuffer:
    def __init__(self, capacity, level=1):
        self.buffer = collections.deque(maxlen=capacity)
        self._level = level
        self._capacity = capacity

        assert level in [1,2,3,4], 'Invalid level. Must be 1, 2, or 3.'

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        # i = random.randrange(self.__len__())
        # head = self.buffer[0]
        # self.buffer[0] = self.buffer[i]
        # self.buffer[i] = head

    def sample_numpy(self, batch_size, random_sample=True):
        if random_sample:
            indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        else:
            indices = range(0, len(self.buffer))

        if self._level == 1:
            states, actions, rewards, dones, next_states = \
                zip(*[self.buffer[idx] for idx in indices])

            return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)

        elif self._level == 2:
            inner_states, outer_states, actions, rewards, \
                dones, next_inner_states, next_outer_states = \
                zip(*[self.buffer[idx] for idx in indices])

            return np.array(inner_states), np.array(outer_states, dtype=np.uint8), \
                np.array(actions, dtype=np.uint8), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_inner_states), np.array(next_outer_states, dtype=np.uint8)
        
        elif self._level == 3:
            inner_states, outer_states, actions, rewards, \
                dones, next_inner_states, next_outer_states, tasks = \
                zip(*[self.buffer[idx] for idx in indices])

            return np.array(inner_states), np.array(outer_states, dtype=np.uint8), \
                np.array(actions, dtype=np.uint8), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_inner_states), np.array(next_outer_states, dtype=np.uint8), \
                np.array(tasks, dtype=np.uint8)
        
        elif self._level == 4:
            inner_states, outer_states, actions, a_distributions, rewards, \
                dones, next_inner_states, next_outer_states = \
                zip(*[self.buffer[idx] for idx in indices])

            return np.array(inner_states), np.array(outer_states, dtype=np.uint8), \
                np.array(actions, dtype=np.uint8), \
                np.array(a_distributions, dtype=np.float32), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_inner_states), np.array(next_outer_states, dtype=np.uint8)

    
    def sample(self, batch_size, random_sample=True, to_torch=True, dev_name='cuda'):
        if self._level == 1:
            states, actions, rewards, dones, next_states = \
                self.sample_numpy(batch_size, random_sample)
            
            if to_torch:
                device = torch.device(dev_name)
                states_th = torch.FloatTensor(states).to(device)
                actions_th = torch.FloatTensor(actions).to(device)
                rewards_th = torch.FloatTensor(rewards).view(-1,1).to(device)
                dones_th = torch.ByteTensor(dones).view(-1,1).float().to(device)
                next_states_th = torch.FloatTensor(next_states).to(device)            
                return states_th, actions_th, rewards_th, dones_th, next_states_th 

            else:
                return states, actions, rewards, dones, next_states 
        
        elif self._level == 2:
            inner_states, outer_states, actions, rewards, \
                dones, next_inner_states, next_outer_states = \
                self.sample_numpy(batch_size, random_sample)
            
            if to_torch:
                device = torch.device(dev_name)
                inner_states_th = torch.FloatTensor(inner_states).to(device)
                outer_states = outer_states.astype(np.float)/255.
                outer_states_th = torch.FloatTensor(outer_states).to(device)
                rewards_th = torch.FloatTensor(rewards).view(-1,1).to(device)
                dones_th = torch.ByteTensor(dones).view(-1,1).float().to(device)
                next_inner_states_th = torch.FloatTensor(next_inner_states).to(device)
                next_outer_states = next_outer_states.astype(np.float)/255.
                next_outer_states_th = torch.FloatTensor(next_outer_states).to(device)
                return inner_states_th, outer_states_th, actions.astype('int'), rewards_th, \
                    dones_th, next_inner_states_th, next_outer_states_th
            
            else:
                return inner_states, outer_states, actions, rewards, \
                    dones, next_inner_states, next_outer_states
        
        elif self._level == 3:
            inner_states, outer_states, actions, rewards, \
                dones, next_inner_states, next_outer_states, tasks = \
                self.sample_numpy(batch_size, random_sample)
            
            if to_torch:
                device = torch.device(dev_name)
                inner_states_th = torch.FloatTensor(inner_states).to(device)
                outer_states = outer_states.astype(np.float)/255.
                outer_states_th = torch.FloatTensor(outer_states).to(device)
                rewards_th = torch.FloatTensor(rewards).view(-1,1).to(device)
                dones_th = torch.ByteTensor(dones).view(-1,1).float().to(device)
                next_inner_states_th = torch.FloatTensor(next_inner_states).to(device)
                next_outer_states = next_outer_states.astype(np.float)/255.
                next_outer_states_th = torch.FloatTensor(next_outer_states).to(device)
                return inner_states_th, outer_states_th, actions.astype('int'), rewards_th, \
                    dones_th, next_inner_states_th, next_outer_states_th, tasks.astype('int')
            
            else:
                return inner_states, outer_states, actions, rewards, \
                    dones, next_inner_states, next_outer_states, tasks
        

        elif self._level == 4:
            inner_states, outer_states, actions, a_distributions, rewards, \
                dones, next_inner_states, next_outer_states = \
                self.sample_numpy(batch_size, random_sample)
            
            if to_torch:
                device = torch.device(dev_name)
                inner_states_th = torch.FloatTensor(inner_states).to(device)
                outer_states = outer_states.astype(np.float)/255.
                outer_states_th = torch.FloatTensor(outer_states).to(device)
                a_distributions_th = torch.FloatTensor(a_distributions).to(device)
                rewards_th = torch.FloatTensor(rewards).view(-1,1).to(device)
                dones_th = torch.ByteTensor(dones).view(-1,1).float().to(device)
                next_inner_states_th = torch.FloatTensor(next_inner_states).to(device)
                next_outer_states = next_outer_states.astype(np.float)/255.
                next_outer_states_th = torch.FloatTensor(next_outer_states).to(device)
                return inner_states_th, outer_states_th, actions.astype('int'), a_distributions_th, rewards_th, \
                    dones_th, next_inner_states_th, next_outer_states_th
            
            else:
                return inner_states, outer_states, actions, a_distributions, rewards, \
                    dones, next_inner_states, next_outer_states
