import numpy as np
from DRIM import System
import os
import pickle

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 130
last_iter = 0
tr_epsds = 4000

modify_memory = False
new_memory_limit = 400000
upper_level_period = 3

# env_names = [   
#             'AntStraightLine-v3', 
#             'AntRotateClock-v3',
#             'AntRotateAntiClock-v3',
#             'AntSquareTrackClock-v3',
#             'AntSquareTrackAntiClock-v3', 
#             'AntGather-v3'
#             ]
# env_names = [   
#             'AntSquareTrack-v3',
#             'AntSquareTrackBomb-v3', 
#             'AntSquareTrackReward-v3'
#             ]
# env_names = [
#             'AntLeft-v3',
#             'AntRight-v3'
#             ]
# env_names = [
#             'AntRotateClock-v3'
#             ]
# env_names = ['Hopper-v2']
env_names = ['AntSquareTrack-v3']

# folder_name = 'HopperTest'
# folder_name = 'AntLeftRight_v2'
# folder_name = 'AntDRIM_v3'
folder_name = 'AntSquareTrackTest'

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
        system.load(common_path, specific_path)
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
    env_steps = 1
    seed = 1000
    params = {
                'env_names': env_names,
                'env_steps': env_steps,
                'init_steps': 1000,
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
                    'n_m_actions': 8, 
                    'n_m_states': 8,
                    'upper_level_period': upper_level_period,
                    'skill_steps': 9,
                    'p_lr': 3e-4,                                        
                    'alpha': 1.5/reward_scale,
                    'mu': 0.5/reward_scale,
                    'beta': 4/reward_scale,
                    'eta': 0.5/reward_scale,
                    'nu': 4/reward_scale,
                    'zeta': 1e-1/reward_scale,
                    'xi': 2/reward_scale,
                    'alpha_upper_level': 5e-2,
                    'seed': seed,                    
                    'r_lr': 3e-3,
                    'cm_lr': 3e-4,
                    'policy_batch_size': batch_size,
                    'concept_batch_size': 512,                    
                    'reward_learning': 'with_concept',
                    'memory_capacity': 400000,
                    'policy_latent_dim': 0,
                    'inconsistency_metric': 'poly',
                    'body_position': True,
                    'model_update_method': 'distribution'
                }    

    system = System(params, agent_params=agent_params)
    
    last_iter = 0    
    mean_rewards = []
    mean_rewards_goal = []
    metrics = []

epsd_steps = 1000//env_steps

system.train_agent( tr_epsds, epsd_steps, initialization=initialization, iter_=last_iter, rewards=mean_rewards, 
                    goal_rewards=mean_rewards_goal, metrics=metrics, common_path=common_path)