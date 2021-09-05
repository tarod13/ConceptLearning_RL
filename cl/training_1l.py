import numpy as np
from CL_1l import System

import os
import pickle

def exists_folder(f_name):
    return os.path.isdir(f_name)

def set_folders(folder, n_test):
    path = os.path.join(folder,str(n_test))
    if not exists_folder(folder): os.mkdir(folder)
    if not exists_folder(path): os.mkdir(path)
    return path

n_test = 0
last_iter = 0
env_name = 'AntMT-v3'
tr_episodes = 1000
folder = '/home/researcher/Diego/ConceptLearning_RL/saved/L1/'

train = True
eval = True


def run(last_iter, path, env_name, tr_episodes, train, eval, 
        initialization: bool = True, eval_task: int = 0,
        reset_eps: bool = False, load_memory: bool = False):

    if last_iter >= 1:
        params = pickle.load(open(path+'/params.p','rb'))
        agent_params = pickle.load(open(path+'/agent_params.p','rb'))
        if reset_eps:
            agent_params.pop('init_epsilon', None)
        print("Params loaded")
        rewards, metrics = [], []
        
        system = System(params, agent_params=agent_params)
        print("System initialized")
        system.load(path, iter=last_iter, load_memory=load_memory)
        print("Nets loaded")

        initialization = False

    else:        
        params = {
            'env_names': [env_name],
            'tr_epsd': tr_episodes,
            }

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
        rewards, metrics = [], []

    if train:
        system.train_agent(initialization=initialization, storing_path=path, 
                            rewards=rewards, metrics=metrics, iter_0=last_iter)
    if eval:
        system.set_envs()
        rewards, events, metric_vector, epsd_lengths = system.eval_agent_skills(
            eval_epsds=10, explore=False, iter_=0, start_render=True, 
            print_space=True, specific_path='video', max_step=0, task=eval_task)
        np.savetxt(os.path.join(path, 'eval_rewards_'+str(eval_task)+'.txt'), np.array(rewards))
        np.savetxt(os.path.join(path, 'eval_events_'+str(eval_task)+'.txt'), np.array(events))
        np.savetxt(os.path.join(path, 'eval_metrics_'+str(eval_task)+'.txt'), np.array(metric_vector))
        np.savetxt(os.path.join(path, 'eval_lengths_'+str(eval_task)+'.txt'), np.array(epsd_lengths))


if __name__ == "__main__":
    path = set_folders(folder, n_test)
    run(last_iter, path, env_name, tr_episodes, train, eval)