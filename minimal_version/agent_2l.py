import torch
from nets import s_Net

def load_first_level_actor(second_level_a_dim=3, first_level_s_dim=91, first_level_a_dim=8, first_level_actor_lr=3e-4, 
                first_level_actor_path="/home/researcher/Diego/Concept_Learning_Ant/Test/19/102_actor_sl.pt"):
    first_level_actor = s_Net(second_level_a_dim, first_level_s_dim, first_level_a_dim, lr=first_level_actor_lr)
    first_level_actor.load_state_dict(torch.load(first_level_actor_path))
    return first_level_actor

def create_second_level_agent(a_dim=3, device='cuda'):
    first_level_actor = load_first_level_actor(second_level_a_dim=a_dim).to(device)
    second_level_agent = Second_Level_Agent(a_dim, first_level_actor, device)
    return second_level_agent

class Second_Level_Agent:
    def __init__(self, a_dim, first_level_actor, device):        
        self.first_level_actor = first_level_actor
        self._a_dim = a_dim
        self._device = device

if __name__ == "__main__":
    agent = create_second_level_agent()
    print("Successful second level agent creation")