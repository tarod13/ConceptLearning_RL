import numpy as np
import torch
from nets import s_Net, vision_actor_critic_Net

from utils import numpy2torch as np2torch


def load_first_level_actor(second_level_a_dim=3, first_level_s_dim=91, first_level_a_dim=8, first_level_actor_lr=3e-4, 
                first_level_actor_path="/home/researcher/Diego/Concept_Learning_Ant/Test/19/102_actor_sl.pt"):
    first_level_actor = s_Net(second_level_a_dim, first_level_s_dim, first_level_a_dim, lr=first_level_actor_lr)
    first_level_actor.load_state_dict(torch.load(first_level_actor_path))
    first_level_actor._a_dim = first_level_a_dim
    return first_level_actor


def create_second_level_agent(n_actions=3, first_level_s_dim=31, noop_action=True,
                agent_type='vision_actor_critic', device='cuda'):
    first_level_actor = load_first_level_actor(second_level_a_dim=n_actions).to(device)
    if agent_type == 'vision_actor_critic':
        second_level_architecture = vision_actor_critic_Net(first_level_s_dim, n_actions+int(noop_action)).to(device)
    else:
        raise RuntimeError('Unkown agent type')
    second_level_agent = Second_Level_Agent(n_actions, second_level_architecture, first_level_actor, device, noop_action)
    return second_level_agent


class Second_Level_Agent:
    def __init__(self, n_actions, second_level_architecture, first_level_actor, device, noop_action):        
        
        self.second_level_architecture = second_level_architecture
        self.first_level_actor = first_level_actor

        self._n_actions = n_actions + int(noop_action)
        self._first_level_a_dim = self.first_level_actor._a_dim
        self._noop = noop_action
        self._device = device
    
    def sample_action(self, state):
        inner_state_np = state['inner_state']
        outer_state_np = state['outer_state']
        first_level_obs_np = state['first_level_obs']
        inner_state, outer_state = np2torch(inner_state_np), np2torch(outer_state_np)
        first_level_obs = np2torch(first_level_obs_np)

        with torch.no_grad():
            A = self.second_level_architecture.sample_action(inner_state, outer_state)
            if self._noop and (A == (self._n_actions - 1)):
                a = np.zeros(self._first_level_a_dim)
            else:
                a = self.first_level_actor.sample_action(first_level_obs, A)
            return A, a
        

if __name__ == "__main__":
    agent = create_second_level_agent()
    print("Successful second level agent creation")