import numpy as np

import torch
import torch.nn as nn

from actor_critic_nets import actor_critic_Net
from utils import numpy2torch as np2torch
from utils import time_stamp

def create_first_level_multitask_agent(env, agent_type='multitask',
                agent_architecture='actor_critic', device='cuda', actor_critic_kwargs={}):
    # Identify dimensions
    s_dim = env.observation_space['state'].shape[0]
    t_dim = env.observation_space['task'].shape[0]
    a_dim = env.action_space.shape[0]
    
    # Generate architecture and agent
    if agent_type == 'multitask':
        if agent_architecture == 'actor_critic':
            architecture = actor_critic_Net(s_dim+t_dim, a_dim, **actor_critic_kwargs)
        else:
            raise RuntimeError('Unkown agent architecture')
    else:
        raise RuntimeError('Unkown agent type')
    first_level_multitask_agent = First_Level_Multitask_Agent(s_dim, t_dim, a_dim, architecture, agent_type).to(device)

    return first_level_multitask_agent


class First_Level_Multitask_Agent(nn.Module):
    def __init__(self, s_dim, t_dim, a_dim, architecture, type):  
        super().__init__()
        self.architecture = architecture 
        self._s_dim = s_dim
        self._t_dim = t_dim
        self._a_dim = a_dim       
        self._type = type
        assert type in ['multitask'], 'Invalid agent type.'
        self._id = time_stamp()
    
    def forward(self, observations):
        pass 
    
    def sample_action(self, observation, explore=True):
        full_state = self.observe_state(observation)
        with torch.no_grad():
            action = self.architecture.sample_action(full_state, explore=explore)            
            return action
    
    def observe_state(self, observation):
        state = observation['state']
        task = observation['task']
        if self._type == 'multitask':
            full_state_np = np.concatenate((state,task))
            full_state = np2torch(full_state_np)
        else:
            raise RuntimeError('Undefined case. TODO or invalid type.')
        return full_state

    def save(self, save_path, best=False):
        if best:
            model_path = save_path + 'best_agent_1l_mt_' + self._id
        else:
            model_path = save_path + 'last_agent_1l_mt_' + self._id
        torch.save(self.state_dict(), model_path)
    
    def load(self, load_directory_path, model_id, device='cuda'):
        dev = torch.device(device)
        self.load_state_dict(torch.load(load_directory_path + 'agent_1l_mt_' + model_id, map_location=dev))



if __name__ == "__main__":
    import gym
    env = gym.make('PendulumMT-v0')
    agent = create_first_level_multitask_agent(env)
    state = env.reset()
    action = agent.sample_action(state)
    full_state = agent.observe_state(state)
    output = agent.architecture(full_state.view(1,-1))
    next_state, reward, done, info = env.step(action)
    print("Successful first level multi-task agent creation")