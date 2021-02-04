import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from policy_optimizers import Optimizer
from utils import one_hot_embedding

class SA_ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=32, clip_value=1.0, 
                lr=3e-4
                ):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.lr = lr
        
    def optimize(self, agent, database, IAs_T_min=1.36): 
        if database.__len__() < self.batch_size:
            return None

        # Set alias for concept network with state and action classifier
        concept_net = agent.concept_architecture

        # Sample batch
        if agent.first_level_agent._type == 'multitask':
            observations, actions, rewards, dones, next_observations = \
                database.sample(self.batch_size)
            
            PS_sT, log_PS_sT, PA_sT, log_PA_sT = agent(observations)
           
            NAST = torch.einsum('ij,ik->jk', PS_sT, PA_sT).detach()
            
            PS_T = NAST.sum(1) / NAST.sum(1).sum(0)
            PA_ST = NAST / NAST.sum(1, keepdim=True)
            PA_T = NAST.sum(0) / NAST.sum(0).sum(0)

            log_PS_T = torch.log(PS_T+1e-10)
            log_PA_T = torch.log(PA_T+1e-10)
            log_PA_ST = torch.log(PA_ST + 1e-10)            
            
            HS_T = -(PS_sT * log_PS_T.unsqueeze(0)).sum(1).mean()
            HS_sT = -(PS_sT * log_PS_sT).sum(1).mean()
            ISs_T = HS_T - HS_sT
            
            HA_T = -(PA_sT * log_PA_T.unsqueeze(0)).sum(1).mean()
            HA_sT = -(PA_sT * log_PA_sT).sum(1).mean()
            HA_ST_state = -((log_PA_ST.unsqueeze(0) * PA_sT.unsqueeze(1).detach()).sum(2) * PS_sT).sum(1).mean()  
            HA_ST_action = -((log_PA_ST.unsqueeze(0) * PA_sT.unsqueeze(1)).sum(2) * PS_sT.detach()).sum(1).mean()  
            
            IAS_T = HA_T.detach() - HA_ST_state
            state_classifier_loss = -IAS_T

            # Optimize state classifier
            concept_net.state_net.classifier.optimizer.zero_grad()
            state_classifier_loss.backward()
            clip_grad_norm_(concept_net.state_net.classifier.parameters(), self.clip_value)
            concept_net.state_net.classifier.optimizer.step()

            IAs_T = HA_T - HA_sT          
            IAS_T_action = HA_T - HA_ST_action
            IAs_ST = IAs_T - IAS_T_action
            alpha = concept_net.log_alpha.exp().item()
            action_classifier_loss = IAs_ST - alpha * IAs_T
            
            # Optimize action classifier
            concept_net.action_net.classifier.optimizer.zero_grad()
            action_classifier_loss.backward()
            clip_grad_norm_(concept_net.action_net.classifier.parameters(), self.clip_value)
            concept_net.action_net.classifier.optimizer.step() 

            alpha_loss = concept_net.log_alpha * (IAs_T - IAs_T_min).detach() 
            
            # Optimize temperature parameter
            concept_net.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([concept_net.log_alpha], self.clip_value)
            concept_net.alpha_optimizer.step()


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
                'state_loss': state_classifier_loss.item(),
                'action_loss': action_classifier_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'alpha': alpha,
            }
            
            return metrics


class S_ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=64, beta=0.0, n_batches_estimation=2, update_rate=0.05, clip_value=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.beta = beta
        self.n_batches_estimation = n_batches_estimation
        self.update_rate = update_rate
        self.PAST = None
        
    def optimize(self, agent, database, n_actions, n_tasks): 
        if database.__len__() < self.batch_size:
            return None

        # # Estimate visits
        # NAST = None

        # with torch.no_grad():
        #     for batch in range(0, self.n_batches_estimation*8):
        #         # Sample batch        
        #         inner_states, outer_states, actions, rewards, dones, \
        #             next_inner_states, next_outer_states, tasks = \
        #             database.sample(self.batch_size)
                
        #         PS_s, log_PS_s = agent(inner_states, outer_states)
        #         A_one_hot = one_hot_embedding(actions, n_actions)
        #         T_one_hot = one_hot_embedding(tasks, n_tasks)

        #         PAS = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
        #         NAST_batch = torch.einsum('ijk,ih->hjk', PAS, T_one_hot).detach() + 1e-8

        #         if NAST is None:
        #             NAST = NAST_batch
        #         else:
        #             NAST = NAST + NAST_batch

        # Sample batch        
        inner_states, outer_states, actions, rewards, dones, \
            next_inner_states, next_outer_states, tasks = \
            database.sample(self.batch_size)
        
        PS_s, log_PS_s = agent(inner_states, outer_states)
        A_one_hot = one_hot_embedding(actions, n_actions)
        T_one_hot = one_hot_embedding(tasks, n_tasks)

        PAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
        NAST = torch.einsum('ijk,ih->hjk', PAS_batch, T_one_hot).detach() + 1e-8
        PAST_batch = NAST / NAST.sum()

        if self.PAST is None:
            self.PAST = PAST_batch
        else:
            self.PAST = self.PAST * (1.-self.update_rate) + PAST_batch * self.update_rate

        # PAST = NAST / NAST.sum()
        PT = self.PAST.sum((1,2))
        PST = self.PAST.sum(2)
        PS_T = PST / PT.view(-1,1)
        PA_ST = self.PAST / PST.unsqueeze(2)
        PAT = self.PAST.sum(1)
        PA_T = PAT / PT.view(-1,1)
        PAS_T = self.PAST / PT.view(-1,1,1)

        log_PS_T = torch.log(PS_T)
        log_PA_T = torch.log(PA_T)
        log_PA_ST = torch.log(PA_ST)         
        
        HS_gT = torch.einsum('ij,hj->ih', PS_s, -log_PS_T).mean(0)
        HS_s = -(PS_s * log_PS_s).sum(1).mean()
        ISs_gT = HS_gT - HS_s
        ISs_T = (PT * ISs_gT).sum()
        
        HA_gT = -(PA_T * log_PA_T).sum(1)
        HA_T = (PT * HA_gT).sum()
        HA_sT = 0.03*np.log(n_actions)
        HA_ST = -(PS_s * log_PA_ST[tasks,:,actions]).sum(1).mean()
        HA_SgT = -(PAS_T * log_PA_ST).sum((1,2))  

        PS_s.unsqueeze(1)

        IAs_gT = HA_gT - HA_sT   
        IAS_gT = HA_gT - HA_SgT
        IAs_SgT = IAs_gT - IAS_gT

        IAs_T = (PT * IAs_gT).sum()
        IAS_T = HA_T - HA_ST
        IAs_ST = IAs_T - IAS_T
         
        
        n_concepts = PS_s.shape[1]
        H_max = np.log(n_concepts)
        classifier_loss = IAs_ST + self.beta * ISs_T            

        agent.classifier.optimizer.zero_grad()
        classifier_loss.backward()
        clip_grad_norm_(agent.classifier.parameters(), self.clip_value)
        agent.classifier.optimizer.step()

        joint_metrics = {
            'HS_T': HS_gT.mean().item(),
            'HS_s': HS_s.item(),
            'HA_T': HA_T.item(),
            'HA_sT': HA_sT,
            'HA_ST': HA_ST.item(),
            'ISs_T': ISs_T.item(),
            'IAs_T': IAs_T.item(),
            'IAS_T': IAS_T.item(),
            'IAs_ST': IAs_ST.item(),
            'loss': classifier_loss.item(),
        }

        metrics_per_task = {}
        for task in range(0, n_tasks):
            metrics_per_task['HS_T'+str(task)] = HS_gT[task].item()
            metrics_per_task['HA_T'+str(task)] = HA_gT[task].item()
            metrics_per_task['HA_ST'+str(task)] = HA_SgT[task].item()
            metrics_per_task['ISs_T'+str(task)] = ISs_gT[task].item()
            metrics_per_task['IAs_T'+str(task)] = IAs_gT[task].item()
            metrics_per_task['IAS_T'+str(task)] = IAS_gT[task].item()
            metrics_per_task['IAs_ST'+str(task)] = IAs_SgT[task].item()

        metrics = {**joint_metrics, **metrics_per_task}
        
        return metrics