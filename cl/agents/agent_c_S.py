import numpy as np

import torch
import torch.nn as nn

from policy_nets import s_Net
from agent_1l_mt import create_first_level_multitask_agent
from concept_nets import *
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp


class Conceptual_Agent(visual_S_concept_Net):
    def __init__(self, s_dim, latent_dim, n_concepts, noisy=True, lr=1e-4):  
        super().__init__(s_dim, latent_dim, n_concepts, noisy, lr)    
        self._id = time_stamp()
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path + 'agent_c_S_' + self._id)
    
    def load(self, load_directory_path, model_id, device='cuda'):
        dev = torch.device(device)
        self.load_state_dict(torch.load(load_directory_path + 'agent_c_S_' + model_id, map_location=dev))


if __name__ == "__main__":
    agent = Conceptual_Agent(31, 512, 10, False)
    print("Successful conceptual agent creation")