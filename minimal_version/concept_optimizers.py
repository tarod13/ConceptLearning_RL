import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from policy_optimizers import Optimizer


class State_ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=32, clip_value=1.0, 
                lr=3e-4
                ):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.lr = lr
        
    def optimize(self, agent, database): 
        if database.__len__() < self.batch_size:
            return None

        # Sample batch
        if agent._type == 'multitask':
            observations, actions, rewards, dones, next_observations = \
                database.sample(self.batch_size)
            
            PS_sT, log_PS_sT, PA_sT, log_PA_sT = agent(observations)
           
            NAST = torch.einsum('ij,ik->jk',PS_sT, PA_sT).detach()
            
            PS_T = NAST.sum(1) / NAST.sum(1).sum(0, keepdim=True)
            PA_ST = NAST / NAST.sum(1, keepdim=True)
            PA_T = NAST.sum(0) / NAST.sum(0).sum(0, keepdim=True)

            log_PS_T = torch.log(PS_T+1e-10)
            log_PA_T = torch.log(PA_T+1e-10)
            log_PA_ST = torch.log(PA_ST + 1e-10)            
            
            HS_T = -(PS_T * log_PS_T).sum(1).mean()
            HS_sT = -(PS_sT * log_PS_sT).sum(1).mean()
            ISs_T = HS_T - HS_sT
            
            HA_T = -(PA_T * log_PA_T).sum(1).mean()
            HA_sT = -(PA_sT * log_PA_sT).sum(1).mean()
            HA_ST = -((log_PA_ST.unsqueeze(0) * PA_sT.unsqueeze(1)).sum(2) * PS_sT).sum(1).mean()  

            IAs_T = HA_T - HA_sT          
            IAS_T = HA_T - HA_ST
            IAs_ST = IAs_T - IAS_T 
            
            classifier_loss = -IAs_ST            

            optimizer = optim.Adam(agent.parameters(), lr=self.lr)  
            optimizer.zero_grad()
            classifier_loss.backward()
            clip_grad_norm_(agent.parameters(), self.clip_value)
            optimizer.step()

            metrics = {
                'HS_T': HS_T.item(),
                'HS_sT': HS_T.item(),
                'HA_T': HS_T.item(),
                'HA_sT': HS_T.item(),
                'HA_ST': HS_T.item(),
                'ISs_T': ISs_T.item(),
                'IAs_T': IAs_T.item(),
                'IAS_T': IAS_T.item(),
                'IAs_ST': IAs_ST.item(),
            }
            
            return metrics