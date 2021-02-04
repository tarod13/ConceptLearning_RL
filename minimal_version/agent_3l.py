import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.distributions import Categorical

from policy_nets import s_Net
from actor_critic_nets import discrete_vision_actor_critic_Net
from agent_2l import load_first_level_actor, Second_Level_Agent
from agent_c_S import Conceptual_Agent
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp, updateNet



def load_conceptual_agent(inner_state_dim, vision_latent_dim, n_concepts, noisy, load_directory_path, model_id):
    conceptual_agent = Conceptual_Agent(inner_state_dim, vision_latent_dim, n_concepts, noisy)
    if model_id is not None:
        conceptual_agent.load(load_directory_path, model_id)
    return conceptual_agent

def create_third_level_agent(load_path_c, model_c_id, n_concepts=20, n_actions=3, first_level_s_dim=33, latent_dim=64,
    n_heads=2, init_log_alpha=0.0, noop_action=True, device='cuda', noisy=False, parallel=True, lr=1e-4, lr_alpha=1e-4, lr_actor=1e-4,
    min_entropy_factor=0.1, lr_c=1e-4, lr_Alpha=1e-4, entropy_update_rate=0.05, target_update_rate=5e-3, init_Epsilon=1.0, 
    delta_Epsilon=7.5e-4):

    first_level_actor = load_first_level_actor(second_level_a_dim=n_actions)
    
    second_level_architecture = discrete_vision_actor_critic_Net(first_level_s_dim, n_actions+int(noop_action),
                                                    latent_dim, n_heads, init_log_alpha, parallel, lr, lr_alpha, lr_actor)

    conceptual_architecture = load_conceptual_agent(first_level_s_dim, latent_dim, n_concepts, noisy, load_path_c, model_c_id)

    third_level_agent = Third_Level_Agent(n_concepts, n_actions, conceptual_architecture, second_level_architecture, 
        first_level_actor, noop_action, min_entropy_factor, lr, lr_Alpha, entropy_update_rate, target_update_rate, init_Epsilon,
        delta_Epsilon).to(device)
    
    return third_level_agent



class Third_Level_Agent(Second_Level_Agent):
    def __init__(self, n_concepts, n_actions, concept_architecture, second_level_architecture, first_level_actor, noop_action, 
        min_entropy_factor=0.1, lr=1e-4, lr_Alpha=1e-4, entropy_update_rate=0.05, target_update_rate=5e-3, init_Epsilon=1.0, 
        delta_Epsilon=7.5e-4, temporal_ratio=5):  
        super().__init__(n_actions, second_level_architecture, first_level_actor, noop_action, temporal_ratio)    
        
        self.concept_architecture = concept_architecture
        freeze(self.concept_architecture)

        self.Q_table = Parameter(torch.Tensor(n_concepts, self._n_actions), requires_grad=False)
        nn.init.constant_(self.Q_table, 0.0)
        self.Q_target = Parameter(torch.Tensor(n_concepts, self._n_actions), requires_grad=False)
        nn.init.constant_(self.Q_target, 0.0)
        self.C_table = Parameter(torch.Tensor(n_concepts, self._n_actions), requires_grad=False)
        nn.init.constant_(self.C_table, 0.0)
        self.Pi_table = Parameter(torch.Tensor(n_concepts, self._n_actions), requires_grad=False)
        nn.init.constant_(self.Pi_table, 1.0/self._n_actions)
        # self.Q_table2 = Parameter(torch.Tensor(n_concepts, self._n_actions))
        # nn.init.constant_(self.Q_table2, 0.0)
        # self.Q_table1_target = Parameter(torch.Tensor(n_concepts, self._n_actions), requires_grad=False)
        # nn.init.constant_(self.Q_table1_target, 0.0)
        # self.Q_table2_target = Parameter(torch.Tensor(n_concepts, self._n_actions), requires_grad=False)
        # nn.init.constant_(self.Q_table2_target, 0.0)
        self.log_Alpha = Parameter(torch.Tensor(1), requires_grad=False)
        nn.init.constant_(self.log_Alpha, 1.0)
        self.Epsilon = Parameter(torch.Tensor(1), requires_grad=False)
        nn.init.constant_(self.Epsilon, init_Epsilon)
        self.H_mean = Parameter(torch.Tensor(1), requires_grad=False)
        nn.init.constant_(self.H_mean, -1.0)
        
        self._n_concepts = n_concepts
        self.H_min = np.log(self._n_actions)
        self.min_Epsilon = min_entropy_factor
        self.delta_Epsilon = delta_Epsilon
        self.lr_Alpha = lr_Alpha
        self.entropy_update_rate = entropy_update_rate
        self.target_update_rate = target_update_rate

        # self.Q_optimizer = Adam([self.Q_table1, self.Q_table2], lr=lr)
        
    
    def save(self, save_path, best=False):
        if best:
            model_path = save_path + 'best_agent_3l_' + self._id
        else:
            model_path = save_path + 'last_agent_3l_' + self._id
        torch.save(self.state_dict(), model_path)
    
    def load(self, load_directory_path, model_id, device='cuda'):
        dev = torch.device(device)
        self.load_state_dict(torch.load(load_directory_path + 'agent_3l_' + model_id, map_location=dev))
    
    # def PA_S(self, target=True):
    #     #Q_min = torch.min(self.Q_table1, self.Q_table2)
    #     if not target:
    #         Q = self.Q_table
    #     else:
    #         Q = self.Q_target
    #     Alpha = self.log_Alpha.exp().item()
    #     Z = torch.logsumexp(Q/(Alpha + 1e-6), dim=1, keepdim=True)
    #     log_PA_S = Q/Alpha - Z
    #     PA_S = log_PA_S.exp() + 1e-6
    #     PA_S = PA_S / PA_S.sum(1, keepdim=True)
    #     log_PA_S = torch.log(PA_S)
    #     return PA_S, log_PA_S

    def PA_S(self):
        PA_S = self.Pi_table
        log_PA_S = torch.log(PA_S)
        return PA_S, log_PA_S
    
    def update_Alpha(self, HA_S): #, n_updates=1):
        # for update in range(0,n_updates):
        #     PA_S, log_PA_S = self.PA_S()
        #     HA_gS = -(PA_S * log_PA_S).sum(1)
        #     HA_S = (PS * HA_gS).sum()
        error = HA_S.item() - self.H_min * self.Epsilon
        new_log_Alpha = self.log_Alpha - self.lr_Alpha * error
        self.log_Alpha.copy_(new_log_Alpha)

        new_Epsilon = torch.max(self.Epsilon - self.delta_Epsilon , self.min_Epsilon * torch.ones_like(self.Epsilon))
        self.Epsilon.copy_(new_Epsilon)
    
    def update_mean_entropy(self, H):
        if self.H_mean < 0.0:
            self.H_mean.copy_(H.detach())
        else:
            H_new = self.H_mean * (1.0-self.entropy_update_rate) + H * self.entropy_update_rate
            self.H_mean.copy_(H_new.detach())
    
    def update_Q(self, Q, C):
        self.Q_table.copy_(Q) 
        self.C_table.copy_(C)
    
    def update_policy(self, Pi):
        self.Pi_table.copy_(Pi)
            
    def update_target(self, rate=1.0):
        Q_target = self.Q_target * (1.-rate) + self.Q_table * rate 
        self.Q_target.copy_(Q_target)  

    def sample_action_from_concept(self, state, explore=True):
        inner_state, outer_state = self.observe_second_level_state(state)
        with torch.no_grad():
            PS_s = self.concept_architecture(inner_state.view(1,-1), outer_state.unsqueeze(0))[0]
            concept = PS_s.argmax(1).item()
            dist = self.Pi_table[concept,:]
            if explore:
                action = Categorical(probs=dist).sample().item()
            else:
                tie_breaking_dist = torch.isclose(dist, dist.max()).float()
                tie_breaking_dist /= tie_breaking_dist.sum()
                action = Categorical(probs=tie_breaking_dist).sample().item()  
            return action, dist.detach().cpu().numpy()
            
    # def update_targets(self):
    #     Q1 = self.Q_table1.detach()
    #     Q2 = self.Q_table2.detach()
    #     Q1_target = self.Q_table1_target.detach()
    #     Q2_target = self.Q_table2_target.detach()
    #     new_target_Q1 = self.target_update_rate * Q1 + (1-self.target_update_rate) * Q1_target
    #     new_target_Q2 = self.target_update_rate * Q2 + (1-self.target_update_rate) * Q2_target
    #     self.Q_table1_target.copy_(new_target_Q1) 
    #     self.Q_table2_target.copy_(new_target_Q2) 
            


if __name__ == "__main__":
    MODEL_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
    concept_path = MODEL_PATH + 'concept_models/'
    concept_model_ID = '2021-01-28_20-23-27'
    agent = create_third_level_agent(concept_path, concept_model_ID)
    print("Successful third level agent creation")