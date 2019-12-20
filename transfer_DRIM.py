import numpy as np
from DRIM import System
import os
import pickle
import torch.optim as optim

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 100
tr_epsds = 4000
epsd_steps = 2000
eval_epsds = 10

origin_n_test = 106
origin_last_iter = 100

env_names = [   
            'AntCrossMaze-v3', 
            ]

origin_folder_name = 'AntDRIM_v5'
folder_name = 'AntDRIMTransfer'

origin_common_path = origin_folder_name + '/' + str(origin_n_test)
origin_specific_path = origin_common_path + '/' + str(origin_last_iter)
common_path = folder_name + '/' + str(n_test)
if not exists_folder(folder_name):
    os.mkdir(folder_name)
if not exists_folder(common_path):
    os.mkdir(common_path)

params = pickle.load(open(origin_common_path+'/params.p','rb'))
agent_params = pickle.load(open(origin_common_path+'/agent_params.p','rb'))
params['env_names'] = env_names
params['render'] = True
params['max_episode_steps'] = epsd_steps
agent_params['n_tasks'] = len(env_names)
# agent_params['cancel_beta_upper_level'] = True
agent_params['alpha_upper_level'] = 1e-2 * np.ones([len(env_names), agent_params['n_m_states']])
# agent_params['beta_upper_level'] = 0.0
# agent_params['alpha'] = 1.0
print("Params loaded")        
system = System(params, agent_params=agent_params)
print("System initialized")
system.load(origin_common_path, origin_specific_path, load_memory=False, load_upper_memory=False, transfer=True)
print("Nets loaded")
print("Finished transfer")

system.train_agent( tr_epsds, epsd_steps, initialization=False, iter_=0, common_path=common_path, learn_lower=False, eval_epsds=eval_epsds, transfer=True)