import numpy as np
from DRIM import System
import os
import pickle
import torch.optim as optim

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 134
last_iter = 0
tr_epsds = 4000

modify_memory = False
new_memory_limit = 400000
upper_level_period = 5
load_upper_memory = True

# env_names = [   
#             'AntGatherRewards-v3', 
#             'AntGatherBombs-v3'
#             ]
env_names = [   
            'AntStraightLine-v3',
            'AntSquareTrack-v3',
            'AntRotate-v3',
            'AntRotateClock-v3',
            'AntGatherRewards-v3', 
            'AntGatherBombs-v3'
            ]
# env_names = [   
#             'AntStraightLine-v3',
#             'AntSquareTrack-v3',
#             'AntRotate-v3',
#             'AntRotateReward-v3',
#             'AntSquareTrackBomb-v3', 
#             'AntSquareTrackReward-v3'
#             ]
# env_names = [
#             'AntLeft-v3',
#             'AntRight-v3'
#             ]
# env_names = ['Hopper-v2']
# env_names = ['AntSquareTrack-v3']

# folder_name = 'HopperTest'
# folder_name = 'AntLeftRight_v2'
folder_name = 'AntDRIM_v5'
# folder_name = 'AntSquareTrackTest'

common_path = folder_name + '/' + str(n_test)
if not exists_folder(folder_name):
    os.mkdir(folder_name)
if not exists_folder(common_path):
    os.mkdir(common_path)

started = False
initialization = True

if last_iter > 0:
    try:
        specific_path = common_path + '/' + str(last_iter)

        params = pickle.load(open(common_path+'/params.p','rb'))
        agent_params = pickle.load(open(common_path+'/agent_params.p','rb'))
        print("Params loaded")        
        mean_rewards = list(np.loadtxt(common_path + '/mean_rewards.txt'))
        mean_rewards_goal = []
        if len(env_names) > 1:
            mean_rewards_goal = list(np.loadtxt(common_path + '/mean_rewards_goal.txt'))
        metrics = list(np.loadtxt(common_path + '/metrics.txt'))
        print("Files loaded")
                
        system = System(params, agent_params=agent_params)
        print("System initialized")
        system.load(common_path, specific_path, load_upper_memory=load_upper_memory)
        print("Nets loaded")

        env_steps = params['env_steps']
        started = True
        initialization = False

        if modify_memory:
            system.agent.memory.capacity = system.agent.params['memory_capacity'] = new_memory_limit
            system.agent.upper_memory.capacity = new_memory_limit // upper_level_period
            system.agent.memory.pointer %= new_memory_limit
            system.agent.upper_memory.pointer %= new_memory_limit // upper_level_period

    except:  
        print("Error loading")      
        started = False  

if not started:
    reward_scale = 10 #10 for (weird?) ant
    batch_size = 256
    n_m_states = 10
    env_steps = 1
    seed = 1000
    params = {
                'env_names': env_names,
                'env_steps': env_steps,
                'init_steps': 1000*len(env_names),
                'seed': seed,
                'beta_coefficient': 1.0,
                'basic_epsds': 0,
                'n_basic_tasks': 0,
                'batch_size': batch_size,
                'render': True
            }
    
    agent_params = {
                    'n_tasks': len(env_names), 
                    'hierarchical': True,
                    'n_m_actions': 4, 
                    'n_m_states': n_m_states,
                    'upper_level_period': upper_level_period,
                    'skill_steps': 9,
                    'p_lr': 3e-4,                                        
                    'alpha': 5.0e-2 * np.ones(len(env_names)),                    
                    'mu': 1.0e-1 * np.ones(len(env_names)),
                    'eta_PS': 0.1 * np.ones(n_m_states),
                    'beta_Ss': 0.1, 
                    'beta_SR': 0.1, 
                    'beta_nSSA': 0.1,
                    'beta_AT': 0.1,
                    'zeta': 0.0,                    
                    'alpha_upper_level': 1e-2 * np.ones([len(env_names), n_m_states]),
                    'beta_upper_level': 1e-2,
                    'nu_upper_level': 1e-2 * np.ones([n_m_states]),
                    'seed': seed,                    
                    'r_lr': 3e-4,
                    'cm_lr': 3e-5,
                    'policy_batch_size': batch_size,
                    'concept_batch_size': 256,                    
                    'reward_learning': 'always',
                    'memory_capacity': 400000,
                    'policy_latent_dim': 0,
                    'inconsistency_metric': 'poly',
                    'body_position': False,
                    'model_update_method': 'distribution',
                    'upper_level_annealing' : False,
                    'upper_policy_steps': 30,
                    'evaluation_upper_level_steps': 1,
                    'threshold_entropy_alpha_upper_level': np.log(1.6),
                    'threshold_entropy_beta_upper_level': np.log(3.5),
                    'threshold_entropy_nu_upper_level': np.log(2.3),
                    'max_threshold_entropy_alpha': 0.5,     
                    'min_threshold_entropy_mu': -8.0,
                    'delta_threshold_entropies': 1.6e-5,
                    'delta_upper_level': 5e-2,
                    'eta_upper_level': 1e-1,
                    'mu_upper_level': 1e-0,
                    'rate_delta_upper_level': 1.0,
                    'min_delta_upper_level': 3e-2,
                    'min_eta_upper_level': 3e-2,
                    'min_mu_upper_level': 3e-2,                    
                    'reward_scale': 10,
                    'tau_alpha': 3e-4,
                    'tau_alpha_S': 3e-4, 
                    'tau_mu': 3e-4,
                    'automatic_lower_temperature': True,
                    'PS_min': 1.0/(2.0*n_m_states),
                    'C_0': 0.0,
                    'threshold_entropy_beta_Ss': 0.5,
                    'threshold_entropy_beta_SR': 0.5,
                    'threshold_entropy_beta_nSSA': 0.5,
                    'threshold_entropy_beta_AT': 0.5+np.log(1.6)-0.2,
                    'n_dims_excluded': 2,
                    'SimPLe_distribution_type_encoder': 'discrete',
                    'transition_model': 'conditional'
                }    

    system = System(params, agent_params=agent_params)
    
    last_iter = 0    
    mean_rewards = []
    mean_rewards_goal = []
    metrics = []

epsd_steps = 1000//env_steps

# system.agent.params['cm_lr'] = 3e-5
# system.agent.params['alpha'] *= 0.5
# system.agent.params['mu'] *= 0.5
# system.agent.alpha *= 0.5 
# system.agent.mu *= 0.5 
# system.agent.concept_model.optimizer = optim.Adam(system.agent.concept_model.parameters(), lr=3e-5)
system.train_agent( tr_epsds, epsd_steps, initialization=initialization, iter_=last_iter, rewards=mean_rewards, 
                    goal_rewards=mean_rewards_goal, metrics=metrics, common_path=common_path, eval_epsds=2*len(env_names))