import numpy as np
from CL_1l import System

import os
import pickle

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 20
last_iter_sl = 0
folder = '/home/researcher/Diego/Concept_Learning_Ant/Test'

path = folder + '/' + str(n_test)
if not exists_folder(folder): os.mkdir(folder)
if not exists_folder(path): os.mkdir(path)

started = False
initialization = True
train = True
load_memory = False
eval_agent = True
load_data = False
task = 0
masked_done = False
initialize_eps = True


if last_iter_sl > 0:
    params = pickle.load(open(path+'/params.p','rb'))
    params['seed'] = 1000
    params.pop('render', None)
    params['eval_epsd_interval'] = 20
    params['masked_done'] = masked_done
    agent_params = pickle.load(open(path+'/agent_params.p','rb'))
    agent_params['dims'] = {
                        'init_prop': 2,
                        'last_prop': 93,
                        'init_ext': 2,
                        'last_ext': 93,
                    }
    agent_params.pop('memory_capacity', None)
    agent_params.pop('clip_value', None)
    if initialize_eps:
        agent_params.pop('init_epsilon', None)
    agent_params.pop('min_epsilon', None)
    agent_params.pop('delta_epsilon', None)
    agent_params.pop('init_beta', None)
    agent_params.pop('init_eta', None)
    agent_params['DQL_epsds_target_update'] = 1000
    agent_params.pop('gamma_E', None)
    print("Params loaded")
    rewards, metrics = [], []
    losses, entropies, entropies_2 = [], [], []
    
    print("Files loaded")
            
    system = System(params, agent_params=agent_params)
    print("System initialized")
    system.load(path, iter=last_iter_sl, load_memory=load_memory)
    print("Nets loaded")

    started = True
    initialization = False

if not started:
    env_names = ['AntMT-v3']

    params = {'env_names': env_names,}

    agent_params = {
                    'dims': {
                        'init_prop': 2,
                        'last_prop': 93,
                        'init_ext': 2,
                        'last_ext': 93,
                    }
                }
    
    system = System(params, agent_params=agent_params)
    last_iter = 0
    rewards = []
    metrics = []
    losses = []
    entropies = []

last_iter = last_iter_sl
if train:
    system.train_agent(initialization=initialization, storing_path=path, 
                        rewards=rewards, metrics=metrics, iter_0=last_iter)
if eval_agent:
    system.set_envs()
    rewards, events, metric_vector, epsd_lenghts = system.eval_agent_skills(
        eval_epsds=100, explore=False, iter_=0, start_render=True, 
        print_space=True, specific_path='video', max_step=0, task=task)
    np.savetxt(path + '/eval_rewards_'+str(task)+'.txt', np.array(rewards))
    np.savetxt(path + '/eval_events_'+str(task)+'.txt', np.array(events))
    np.savetxt(path + '/eval_metrics_'+str(task)+'.txt', np.array(metric_vector))
    np.savetxt(path + '/eval_lenghts_'+str(task)+'.txt', np.array(epsd_lenghts))
