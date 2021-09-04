import numpy as np
import torch
import random
import time
import datetime
import yaml


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

def one_hot_embedding(labels, num_classes, device='cuda'):
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

def time_stamp():
    time_in_seconds = time.time()
    stamp = datetime.datetime.fromtimestamp(time_in_seconds).strftime('%Y-%m-%d_%H-%M-%S')
    return stamp

def cat_state_task(observation):
    state = observation['state']
    task = observation['task']
    obs = np.concatenate((state, task))
    return obs

def load_env_model_pairs(file):
    yaml_file = open(file, 'r')
    try:
        env_model_pairs = yaml.load(yaml_file, Loader=yaml.FullLoader)['env_model_pairs']
        assert isinstance(env_model_pairs, dict)
    except:
        raise RuntimeError('Invalid file. It should be a dictionary with name "env_model_pairs"')
    return env_model_pairs