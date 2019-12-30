import numpy as np
from DRIMseq import System

import os
import pickle

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 0
last_iter = 93
folder = 'Test'

path = folder + '/' + str(n_test)
if not exists_folder(folder): os.mkdir(folder)
if not exists_folder(path): os.mkdir(path)

started = False
initialization = True
skill_learning = False

if last_iter > 0:
    # try:
    specific_path = path + '/' + str(last_iter)

    params = pickle.load(open(path+'/params.p','rb'))
    params['env_names_ql'] = [
                                'AntStraightLineRestricted-v3'
                            ]
    # params['env_names_ql'] = [
    #                             'AntStraightLineRestricted-v3',
    #                             'AntSquareTrack-v3',
    #                             'AntGatherRewards-v3', 
    #                             'AntGatherBombs-v3'
    #                         ]
    params['tr_steps_ql'] = 300
    params['env_steps_ql'] = 10
    agent_params = pickle.load(open(path+'/agent_params.p','rb'))
    agent_params['decision_type'] = 'epsilon'
    agent_params['init_epsilon'] = 1.0
    agent_params['min_epsilon'] = 0.1
    agent_params['delta_epsilon'] = 1.6e-6
    agent_params['DQN_learning_type'] = 'DQL'
    agent_params['dim_excluded']['last'] = 60
    agent_params['min_epsilon'] = 0.05
    #alpha_sl = agent_params['alpha']
    #agent_params['alpha'] = {}
    #agent_params['alpha']['sl'] = alpha_sl
    #agent_params['alpha']['ql'] = 1.0
    # agent_params['lr']['ql'] = {
    #                             'q': 3e-4,
    #                             'v': 3e-4,
    #                             'alpha': 3e-4,
    #                             'v_target': 5e-3
    #                         }
    # agent_params['batch_size']['ql'] = 256
    print("Params loaded")        
    rewards = list(np.loadtxt(path + '/mean_rewards.txt'))
    metrics = list(np.loadtxt(path + '/metrics.txt'))
    print("Files loaded")
            
    system = System(params, agent_params=agent_params, skill_learning=skill_learning)
    # system.learning_type = 'ql'
    print("System initialized")
    system.load(path, specific_path)
    print("Nets loaded")

    started = True
    initialization = False

    # except:  
    #     print("Error loading")      
    #     started = False
    #     initialization = True

if not started:
    env_names_sl = [
                    'AntStraightLine-v3',
                    'AntRotate-v3',
                    'AntRotateClock-v3'
                ]

    env_names_ql = [
                    'AntStraightLineRestricted-v3',
                    'AntSquareTrack-v3',
                    'AntGatherRewards-v3', 
                    'AntGatherBombs-v3'
                ]

    params = {
                'seed': 1000,
                'env_names_sl': env_names_sl,
                'env_names_ql': env_names_ql,
                'env_names_tl': [],
                'env_steps': 1, 
                'grad_steps': 1, 
                'init_steps': 1000,
                'max_episode_steps': 1000,
                'batch_size': 256, 
                'render': True, 
                'reset_when_done': True, 
                'store_video': False                            
            }
    
    system = System(params)
    last_iter = 0
    rewards = []
    metrics = []

system.train_agent(initialization=initialization, skill_learning=skill_learning, storing_path=path, rewards=rewards, metrics=metrics, iter_0=last_iter)
