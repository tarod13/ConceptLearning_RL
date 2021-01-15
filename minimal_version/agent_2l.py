import numpy as np

import torch
import torch.nn as nn

from nets import s_Net, vision_actor_critic_Net
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp

def load_first_level_actor(second_level_a_dim=3, first_level_s_dim=91, first_level_a_dim=8, 
                first_level_actor_path="/home/researcher/Diego/Concept_Learning_Ant/Test/19/102_actor_sl.pt"):
    first_level_actor = s_Net(second_level_a_dim, first_level_s_dim, first_level_a_dim)
    first_level_actor.load_state_dict(torch.load(first_level_actor_path))
    first_level_actor._a_dim = first_level_a_dim
    return first_level_actor


def create_second_level_agent(n_actions=3, first_level_s_dim=31, noop_action=True,
                agent_type='vision_actor_critic', device='cuda'):
    first_level_actor = load_first_level_actor(second_level_a_dim=n_actions)
    if agent_type == 'vision_actor_critic':
        second_level_architecture = vision_actor_critic_Net(first_level_s_dim, n_actions+int(noop_action))
    else:
        raise RuntimeError('Unkown agent type')
    second_level_agent = Second_Level_Agent(n_actions, second_level_architecture, first_level_actor, noop_action).to(device)
    return second_level_agent


class Second_Level_Agent(nn.Module):
    def __init__(self, n_actions, second_level_architecture, first_level_actor, noop_action, temporal_ratio=5):  
        super().__init__()    
        
        self.second_level_architecture = second_level_architecture
        self.first_level_actor = first_level_actor
        freeze(self.first_level_actor)

        self._n_actions = n_actions + int(noop_action)
        self._first_level_a_dim = self.first_level_actor._a_dim
        self._noop = noop_action
        self._temporal_ratio = temporal_ratio
        self._id = time_stamp()
    
    def forward(self, states):
        inner_state, outer_state, first_level_obs = self.observe_state(state)
        second_level_output = self.second_level_architecture(inner_state, outer_state)
        first_level_output = self.first_level_actor(first_level_obs)
        return first_level_output, second_level_output 
    
    def sample_action(self, state):
        inner_state, outer_state = self.observe_second_level_state(state)
        with torch.no_grad():
            action = self.second_level_architecture.sample_action(inner_state, outer_state)            
            return action
    
    def sample_first_level_action(self, state, action):
        first_level_obs = self.observe_first_level_state(state)
        with torch.no_grad():
            # If skill selected correspond to the no-operation skill, then
            # fill action vector with zeros. Otherwise, sample from skill
            if self._noop and (action == (self._n_actions - 1)):
                first_level_action = np.zeros(self._first_level_a_dim)
            else:
                first_level_action = self.first_level_actor.sample_action(
                    first_level_obs, action)
            return first_level_action

    def observe_state(self, state):
        inner_state, outer_state = self.observe_second_level_state(state)
        first_level_obs = self.observe_first_level_state(state)
        return inner_state, outer_state, first_level_obs

    def observe_second_level_state(self, state):
        inner_state_np = state['inner_state']
        outer_state_np = state['outer_state']        
        inner_state, outer_state = np2torch(inner_state_np), np2torch(outer_state_np)        
        return inner_state, outer_state
    
    def observe_first_level_state(self, state):
        first_level_obs_np = state['first_level_obs']
        first_level_obs = np2torch(first_level_obs_np)
        return first_level_obs
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path + 'agent_2l_' + self._id)
    
    def load(self, load_directory_path, model_id, device='cuda'):
        dev = torch.device(device)
        self.load_state_dict(torch.load(load_directory_path + 'agent_2l_' + model_id, map_location=dev))


if __name__ == "__main__":
    agent = create_second_level_agent()
    print("Successful second level agent creation")