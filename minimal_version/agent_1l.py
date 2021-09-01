import pickle

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from nets_seq import Memory, v_Net, q_Net, s_Net
from utils import updateNet

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class Agent:
    '''
    '''
    def __init__(self, s_dim, a_dim, n_tasks, params, seed=0):

        self.params = params.copy()
        default_params = {
                            'n_skills': 8,
                            'alpha': 0.1,                                    
                            'init_epsilon': 1.0,
                            'min_epsilon': 0.4,                            
                            'delta_epsilon': 2.5e-7,                            
                            'init_threshold_entropy_alpha': 0.0,
                            'delta_threshold_entropy_alpha': 8e-6,       
                            
                            'lr': {
                                'q': 3e-4,
                                'v': 3e-4,
                                'pi': 3e-4,
                                'alpha': 3e-4,
                                'v_target': 5e-3,
                                },

                            'dims': {
                                        'init_prop': 2,
                                        'last_prop': s_dim,
                                        'init_ext': 3,
                                        'last_ext': s_dim-60
                                    },
                            
                            'batch_size': 256,
                            'memory_capacity': 1200000,
                            'gamma_E': 0.99,
                            'clip_value': 0.5,
                            'entropy_annealing': False,                                                                        
                        }
        
        for key, value in default_params.items():
            if key not in self.params.keys():
                self.params[key] = value
        
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.sa_dim = self.s_dim + self.a_dim
        self.sars_dim = 2*self.s_dim + self.a_dim + 1
        self.sarsd_dim = self.sars_dim + 1
        self.t_dim = self.sarsd_dim + 3

        self.n_tasks = n_tasks
        self.seed = seed
        self.n_skills = n_tasks
        self.counter = 0
        v_dim = n_tasks

        self.dims = self.params['dims']
        self.batch_size = self.params['batch_size']
        self.lr = self.params['lr']
        self.gamma_E = self.params['gamma_E']
        self.clip_value = self.params['clip_value']
        self.entropy_annealing = self.params['entropy_annealing']
        
        self.max_noise = 1e-10
        self.r_mean = 0.0
        self.r_std = 1.0
        self.r_initialized = False
        self.r_max = 0.0

        # Metric weights
        self.min_threshold_entropy_alpha = -a_dim*1.0/2
        self.threshold_entropy_alpha = self.params['init_threshold_entropy_alpha']
        self.delta_threshold_entropy_alpha = self.params['delta_threshold_entropy_alpha']
        self.alpha = self.params['alpha']
        self.epsilon = self.params['init_epsilon']
        self.min_epsilon = self.params['min_epsilon']
        self.delta_epsilon = self.params['delta_epsilon']
        
        # Nets and memory
        self.v = v_Net(self.dims['last_ext']-self.dims['init_ext'], v_dim, lr=self.lr['v']).to(device)
        self.v_target = v_Net(self.dims['last_ext']-self.dims['init_ext'], v_dim, lr=self.lr['v']).to(device)
        self.critic1 = q_Net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr['q']).to(device)
        self.critic2 = q_Net(self.dims['last_ext']-self.dims['init_ext'], a_dim, v_dim, lr=self.lr['q']).to(device)
        self.actor = s_Net(self.n_skills, self.dims['last_prop']-self.dims['init_prop'], a_dim, lr=self.lr['pi']).to(device)        
        self.memory = Memory(self.params['memory_capacity'], n_seed=self.seed)
        
        updateNet(self.v_target, self.v,1.0)

    def memorize(self, event, init=False):
        if init:
            self.memory.store(event[np.newaxis,:])
        else:
            self.memory.store(event.tolist())
    
    def act(self, state, skill, explore=True):
        s_cuda = torch.FloatTensor(state[self.dims['init_prop']:self.dims['last_prop']]).to(device)
        with torch.no_grad():
            a = self.actor.sample_action(s_cuda, skill, explore=explore)if skill < self.n_skills else np.zeros(self.a_dim)
            return a     
                  
               
    def learn_skills(self, only_metrics=False):
        batch = self.memory.sample(self.batch_size)
        batch = np.array(batch)
        batch_size = batch.shape[0]        

        if batch_size > 0:
            s_batch = torch.FloatTensor(batch[:,self.dims['init_ext']:self.dims['last_ext']]).to(device)
            s_batch_prop = torch.FloatTensor(batch[:,self.dims['init_prop']:self.dims['last_prop']]).to(device)
            a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
            r_batch = torch.FloatTensor(batch[:,self.sa_dim]).view(-1,1).to(device)
            ns_batch = torch.FloatTensor(batch[:,self.sa_dim+1+self.dims['init_ext']:self.sa_dim+1+self.dims['last_ext']]).to(device)
            d_batch = torch.FloatTensor(batch[:,self.sars_dim]).view(-1,1).to(device)
            T_batch = batch[:,self.sarsd_dim].astype('int')  

            if not only_metrics:
                # Optimize q networks
                q1_E = self.critic1(s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                q2_E = self.critic2(s_batch, a_batch)[np.arange(batch_size), T_batch].view(-1,1)
                next_v_E = self.v_target(ns_batch)[np.arange(batch_size), T_batch].view(-1,1)               

                q_approx_E = r_batch + self.gamma_E * next_v_E * (1-d_batch)
                
                q1_loss = self.critic1.loss_func(q1_E, q_approx_E.detach())
                self.critic1.optimizer.zero_grad()
                q1_loss.backward()
                clip_grad_norm_(self.critic1.parameters(), self.clip_value)
                self.critic1.optimizer.step()
                
                q2_loss = self.critic2.loss_func(q2_E, q_approx_E.detach())
                self.critic2.optimizer.zero_grad()
                q2_loss.backward()
                clip_grad_norm_(self.critic2.parameters(), self.clip_value)
                self.critic2.optimizer.step()

            # Optimize v network
            a_batch_A, log_pa_sApT_A = self.actor.sample_actions_and_llhoods_for_all_skills(s_batch_prop.detach())
            A_batch = T_batch
            a_batch_off = a_batch_A[np.arange(batch_size), A_batch, :]
            log_pa_sT = log_pa_sApT_A[np.arange(batch_size), A_batch, A_batch].view(-1,1)
            
            q1_off_E = self.critic1(s_batch.detach(), a_batch_off)
            q2_off_E = self.critic2(s_batch.detach(), a_batch_off)
            q_off_E = torch.min(torch.stack([q1_off_E, q2_off_E]), 0)[0][np.arange(batch_size), T_batch].view(-1,1)
            
            v_approx_E = q_off_E - self.alpha * log_pa_sT
            
            if not only_metrics:
                v_E = self.v(s_batch)[np.arange(batch_size), T_batch].view(-1,1)
                
            task_mask = torch.zeros(batch_size, self.n_skills).float().to(device)
            task_mask[np.arange(batch_size), T_batch] = torch.ones(batch_size).float().to(device)
            task_count = task_mask.sum(0).view(-1,1)
            task_mask_distribution = task_mask / (task_count.view(1,-1) + 1e-10)
            Ha_sT = -(log_pa_sT * task_mask_distribution).sum(0)
            if self.entropy_annealing: alpha_gradient = Ha_sT.detach() - self.threshold_entropy_alpha

            if not only_metrics:
                v_loss = self.v.loss_func(v_E.view(-1,1), v_approx_E.view(-1,1).detach()) # + self.v.loss_func(v_I, v_approx_I.detach())
                self.v.optimizer.zero_grad()
                v_loss.backward()
                clip_grad_norm_(self.v.parameters(), self.clip_value)
                self.v.optimizer.step()
                updateNet(self.v_target, self.v, self.lr['v_target'])
                
                # Optimize skill network
                self.actor.optimizer.zero_grad()              
                pi_loss = -(v_approx_E).mean()
                pi_loss.backward()              
                clip_grad_norm_(self.actor.parameters(), self.clip_value)
                self.actor.optimizer.step()                    

                if self.entropy_annealing:
                    # Optimize dual variable                
                    log_alpha = torch.log(self.alpha + 1e-6)
                    log_alpha -= self.lr['alpha'] * alpha_gradient
                    self.alpha = torch.exp(log_alpha).clamp(1e-10, 1e+3)

                    self.threshold_entropy_alpha = np.max([self.threshold_entropy_alpha - self.delta_threshold_entropy_alpha, self.min_threshold_entropy_alpha])
                    
        else:
            log_pa_sT = torch.zeros(1).to(device)  
            Ha_sT = torch.zeros(1).to(device)
            
        if only_metrics:
            metrics = {
                        'H(a|s,T)': Ha_sT.mean().detach().cpu().numpy()                
                    }            
            return metrics    
   
    def estimate_metrics(self):
        metrics = {}
        with torch.no_grad():
            metrics = self.learn_skills(only_metrics=True)
        return metrics
    
    def save(self, common_path, specific_path):
        self.params['alpha'] = self.alpha
        self.params['init_threshold_entropy_alpha'] = self.threshold_entropy_alpha
        self.params['init_epsilon'] = self.epsilon
        pickle.dump(self.params,open(common_path+'/agent_params.p','wb'))

        data_batches = {'l': len(self.memory.data)//20000+1}
        for i in range(0, data_batches['l']):
            if i+1 < data_batches['l']:
                pickle.dump(self.memory.data[20000*i:20000*(i+1)],open(common_path+'/memory_'+str(i+1)+'.p','wb'))
            else:
                pickle.dump(self.memory.data[20000*i:-1],open(common_path+'/memory_'+str(i+1)+'.p','wb'))
        pickle.dump(data_batches,open(common_path+'/data_batches.p','wb'))
        
        torch.save(self.critic1.state_dict(), specific_path+'_critic1.pt')        
        torch.save(self.critic2.state_dict(), specific_path+'_critic2.pt')    
        torch.save(self.v.state_dict(), specific_path+'_v.pt')
        torch.save(self.v_target.state_dict(), specific_path+'_v_target.pt')
        torch.save(self.actor.state_dict(), specific_path+'_actor.pt')
    
    def load(self, common_path, specific_path, load_memory=True):
        if load_memory: 
            data_batches = pickle.load(open(common_path+'/data_batches.p','rb'))
            pointer = 0
            for i in range(0, data_batches['l']):
                try:
                    data = pickle.load(open(common_path+'/memory_'+str(i+1)+'.p','rb'))
                    self.memory.data += data
                    pointer += len(data)
                except:
                    pass
            self.memory.pointer = pointer % self.memory.capacity
            self.memory.data = self.memory.data[-self.memory.capacity:]

        self.v.load_state_dict(torch.load(specific_path+'_v.pt'))
        self.v_target.load_state_dict(torch.load(specific_path+'_v_target.pt'))
        self.v.train()
        self.v_target.train()  

        self.actor.load_state_dict(torch.load(specific_path+'_actor.pt'))
        self.actor.train()
        
        self.critic1.load_state_dict(torch.load(specific_path+'_critic1.pt'))
        self.critic2.load_state_dict(torch.load(specific_path+'_critic2.pt'))                

        self.critic1.train()
        self.critic2.train()                 
   
    def reset_memory(self):
        self.memory.forget()