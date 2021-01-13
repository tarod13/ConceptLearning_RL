# https://github.com/ku2482/sac-discrete.pytorch/blob/42b53f91d30e69b8bc9b415c314fd2c29906856a/code/memory.py
from collections import deque
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler


class MultiStepBuff:
    keys = ["state", "action", "reward", "intrinsic_reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
            }

    def append(self, state, action, reward, intrinsic_reward):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)
        self.memory["intrinsic_reward"].append(intrinsic_reward)

    def get(self, gamma_E=0.999, gamma_I=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma_E, 'e')
        intrinsic_reward = self._multi_step_reward(gamma_I, 'i')
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        _ = self.memory["intrinsic_reward"].popleft()
        return state, action, reward, intrinsic_reward

    def _multi_step_reward(self, gamma, type_):
        if type_ == 'e':
            return np.sum([
                r * (gamma ** i) for i, r
                in enumerate(self.memory["reward"])])
        elif type_ == 'i':
            return np.sum([
                r * (gamma ** i) for i, r
                in enumerate(self.memory["intrinsic_reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class LazyMemory(dict):
    state_keys = ['state', 'next_state']
    np_keys = ['action', 'reward', 'done', 'intrinsic_reward']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        for key in self.state_keys:
            self[key] = []

        self['action'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)        
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['intrinsic_reward'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done, intrinsic_reward,
               episode_done=None):
        self._append(state, action, reward, next_state, done, intrinsic_reward)

    def _append(self, state, action, reward, next_state, done, intrinsic_reward):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done
        self['intrinsic_reward'][self._p] = intrinsic_reward

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.float32)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = np.array(self['action'][indices]).reshape(-1)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)
        intrinsic_rewards = torch.FloatTensor(self['intrinsic_reward'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones, intrinsic_rewards

    def __len__(self):
        return len(self['state'])

    def get(self):
        return dict(self)

    def load(self, memory):
        for key in self.state_keys:
            self[key].extend(memory[key])

        num_data = len(memory['state'])
        if self._p + num_data <= self.capacity:
            for key in self.np_keys:
                self[key][self._p:self._p+num_data] = memory[key]
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            for key in self.np_keys:
                self[key][self._p:] = memory[key][:mid_index]
                self[key][:end_index] = memory[key][mid_index:]

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity
        self.truncate()
        assert self._n == len(self)


class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done, intrinsic_reward,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward, intrinsic_reward)

            if len(self.buff) == self.multi_step:
                state, action, reward, intrinsic_reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, intrinsic_reward)

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(state, action, reward, next_state, done, intrinsic_reward)


class LazyPrioritizedMemory(LazyMultiStepMemory):
    state_keys = ['state', 'next_state']
    np_keys = ['action', 'reward', 'done', 'intrinsic_reward', 'priority']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3, alpha=0.6, beta=0.4, beta_annealing=0.001,
                 epsilon=1e-4):
        super(LazyPrioritizedMemory, self).__init__(
            capacity, state_shape, device, gamma, multi_step)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def reset(self):
        super(LazyPrioritizedMemory, self).reset()
        self['priority'] = np.empty((self.capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, next_state, done, intrinsic_reward, error,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward, intrinsic_reward)

            if len(self.buff) == self.multi_step:
                state, action, reward, intrinsic_reward = self.buff.get(self.gamma)
                self['priority'][self._p] = self.calc_priority(error)
                self._append(state, action, reward, next_state, done, intrinsic_reward)

            if episode_done or done:
                self.buff.reset()
        else:
            self['priority'][self._p] = self.calc_priority(error)
            self._append(
                state, action, reward, next_state, done, intrinsic_reward)

    def update_priority(self, indices, errors):
        self['priority'][indices] = np.reshape(
            self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def get(self):
        state_dict = {key: self[key] for key in self.state_keys}
        np_dict = {key: self[key][:self._n] for key in self.np_keys}
        state_dict.update(**np_dict)
        return state_dict

    def sample(self, batch_size):
        self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(
            self['priority'][:self._n, 0], batch_size)
        indices = list(sampler)

        batch = self._sample(indices, batch_size)
        priorities = np.array(self['priority'][indices], dtype=np.float32)
        priorities = priorities / np.sum(self['priority'][:self._n])

        weights = (self._n * priorities) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(
            weights).view(batch_size, -1).to(self.device)

        return batch, indices, weights