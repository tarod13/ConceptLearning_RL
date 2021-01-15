import collections
import numpy as np
import torch

ExperienceFirstLevel = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])
PixelExperienceSecondLevel = collections.namedtuple(
    'Experience', field_names=['inner_state', 'outer_state', 'action', 'reward', 
                                'done', 'next_inner_state', 'next_outer_state'])

class PixelExperienceBuffer:
    def __init__(self, capacity, level=1):
        self.buffer = collections.deque(maxlen=capacity)
        self._level = level

        assert (level == 1) or (level == 2), 'Invalid level. Must be 1 or 2.'

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample_numpy(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
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

            return np.array(inner_states), np.array(outer_states), \
                np.array(actions, dtype=np.uint8), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_inner_states), np.array(next_outer_states)

    
    def sample(self, batch_size, to_torch=True, dev_name='cuda'):
        if self._level == 1:
            states, actions, rewards, dones, next_states = \
                self.sample_numpy(batch_size)
            
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
                self.sample_numpy(batch_size)
            
            if to_torch:
                device = torch.device(dev_name)
                inner_states_th = torch.FloatTensor(inner_states).to(device)
                outer_states_th = torch.FloatTensor(outer_states).to(device)
                rewards_th = torch.FloatTensor(rewards).view(-1,1).to(device)
                dones_th = torch.ByteTensor(dones).view(-1,1).float().to(device)
                next_inner_states_th = torch.FloatTensor(next_inner_states).to(device)
                next_outer_states_th = torch.FloatTensor(next_outer_states).to(device)
                return inner_states_th, outer_states_th, actions.astype('int'), rewards_th, \
                    dones_th, next_inner_states_th, next_outer_states_th
            
            else:
                return inner_states, outer_states, actions, rewards, \
                    dones, next_inner_states, next_outer_states
