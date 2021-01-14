import numpy as np
import torch
import random
import gym
import collections

def numpy2torch(np_array, device='cuda'):
    return torch.FloatTensor(np_array).to(device)

def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def scale_action(a, min, max):
    return (0.5*(a+1.0)*(max-min) + min)

def vectorized_multinomial(prob_matrix):
    items = np.arange(prob_matrix.shape[1])
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0],1)
    k = (s < r).sum(axis=1)
    return items[k]

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes).to(device) 
    return y[labels]

def set_seed(n_seed=0, device='cuda'):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == 'cuda': torch.cuda.manual_seed(n_seed)

def is_float(x):
    return isinstance(x, float)

def is_tensor(x):
    return isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor)


class AntPixelWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return AntPixelWrapper.separate_state(obs)

    @staticmethod
    def separate_state(obs):
        state = collections.OrderedDict()
        state['inner_state'] = obs['state'][2:-60] # Eliminate the xy coordinates (first 2 entries) and the 'lidar' maze observations
        outer_state = obs['pixels'].astype(np.float) / 255.0
        outer_state = np.swapaxes(outer_state, 1, 2)
        state['outer_state'] = np.swapaxes(outer_state, 0, 1)
        state['first_level_obs'] = obs['state'][2:]
        return state