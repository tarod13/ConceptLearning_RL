import numpy as np
from DRIMseq import System

import os
import pickle

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 13
last_iter_sl = 156
last_iter_ql = 113
last_iter_cl = 50000
last_iter_tl = 0
folder = 'Test'

path = folder + '/' + str(n_test)
if not exists_folder(folder): os.mkdir(folder)
if not exists_folder(path): os.mkdir(path)

started = False
initialization = True
skill_learning = False
q_learning = False
concept_learning = True

if (last_iter_sl + last_iter_ql + last_iter_cl) > 0:
    # try:
    params = pickle.load(open(path+'/params.p','rb'))
    # params['tr_steps_cl'] = 1200000
    # params['env_names_tl'] = [
                            #     'AntCrossMaze-v3'
                            # ]
    # params['env_steps_tl'] = 100
    agent_params = pickle.load(open(path+'/agent_params.p','rb'))
    # agent_params.pop('beta', None)
    # agent_params.pop('lr', None)
    # agent_params.pop('init_threshold_entropy_alpha_cl', None)
    # agent_params['alpha']['cl'] = 1e-2
    # agent_params['lr']['cl'] = {'alpha': 3e-4}
    print("Params loaded")
    if skill_learning or q_learning:  
        suffix = '_sl' if skill_learning else '_ql'
    try:      
        rewards = list(np.loadtxt(path + '/mean_rewards'+suffix+'.txt'))        
    except:
        rewards = []        

    try:      
        metrics = list(np.loadtxt(path + '/metrics'+suffix+'.txt'))        
    except:        
        metrics = []
    
    try:      
        losses = list(np.loadtxt(path + '/concept_training_losses_we.txt')) if last_iter_cl > 0 else []     #TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO   
    except:
        losses = []
        
    try:      
        entropies = list(np.loadtxt(path + '/concept_training_entropies_we.txt')) if last_iter_cl > 0 else []       
    except:
        entropies = []
    
    print("Files loaded")
            
    system = System(params, agent_params=agent_params, skill_learning=skill_learning)
    # system.steps['tr']['cl'] = 1200000
    # system.agent.alpha['cl'] = 1e-2
    # system.env_names['tl'] = [
    #                             'AntCrossMaze-v3'
    #                         ]
    # system.n_tasks['tl'] = 1
    # system.agent.threshold_entropy_alpha['cl'] = np.log(10)
    print("System initialized")
    system.load(path, iter_0_sl=last_iter_sl, iter_0_ql=last_iter_ql, iter_0_cl=last_iter_cl, load_memory=(skill_learning or q_learning or concept_learning))
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

last_iter = last_iter_sl if skill_learning else (last_iter_ql if q_learning else (last_iter_cl if concept_learning else last_iter_tl))
system.train_agent(initialization=initialization, skill_learning=skill_learning, storing_path=path, 
                    rewards=rewards, metrics=metrics, losses=losses, entropies=entropies, 
                    iter_0=last_iter, q_learning=q_learning, concept_learning=concept_learning)
