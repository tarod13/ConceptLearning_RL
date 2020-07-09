import numpy as np
from CL import System # DRIMseq

import os
import pickle

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 17
last_iter_sl = 102 # 102
last_iter_sl_2 = 0
last_iter_ql = 0 # 40
last_iter_cl = 0 # 215000
last_iter_tl = 0 # 86
folder = 'Test'

path = folder + '/' + str(n_test)
if not exists_folder(folder): os.mkdir(folder)
if not exists_folder(path): os.mkdir(path)

started = False
initialization = True
skill_learning = False
q_learning = True
concept_learning = False
transfer_learning = False
train = True
load_memory = False
classify = False
eval_agent = False
load_data = False
task = 0
masked_done = False
active_RND = False
active_MC = False
initialize_eps = True

###

# started = False
# initialization = True
# skill_learning = False
# q_learning = False
# concept_learning = True
# transfer_learning = False
# train = True
# load_memory = True
# classify = False
# eval_agent = False
# load_data = False
# task = 0
# masked_done = False
# active_RND = False
# active_MC = False

######

# started = False
# initialization = True
# skill_learning = False
# q_learning = True
# concept_learning = False
# transfer_learning = False
# train = False
# load_memory = False
# classify = False
# eval_agent = True
# load_data = False
# task = 2
# masked_done = False
# active_RND = False
# active_MC = False

# # ##

# started = False
# initialization = True
# skill_learning = False
# q_learning = False
# concept_learning = False
# transfer_learning = False
# train = False
# load_memory = True
# classify = True
# eval_agent = False
# load_data = False
# task = 2
# if task >= 2:
#     load_memory = False
#     load_data = True
# masked_done = False
# active_RND = False
# active_MC = False

# # ##

# started = False
# initialization = True
# skill_learning = False
# q_learning = True
# concept_learning = False
# transfer_learning = False
# train = True
# load_memory = False
# classify = False
# eval_agent = False
# load_data = False
# task = 0

# started = False
# initialization = False
# skill_learning = False
# q_learning = False
# concept_learning = False
# transfer_learning = True
# train = True
# load_memory = False
# classify = False
# eval_agent = False
# load_data = False
# task = 0


if (last_iter_sl + last_iter_ql + last_iter_cl) > 0:
    # try:
    params = pickle.load(open(path+'/params.p','rb'))
    # params['env_names_sl'] = [
    #                 'AntStraightLineMG-v3'
    #             ]
    # params['env_names_sl'] = [
    #                 'AntStraightLine-v3',
    #                 'AntRotate-v3',
    #                 'AntRotateClock-v3'
    #             ]
    # params['env_names_ql'] = [
    #                 'AntSquareWall-v3',
    #                 'AntSquareTrack-v3',
    #                 'AntGather-v3', 
    #                 'AntAvoid-v3'
    #             ]
    # params['tr_steps_tl'] = 10000
    # params['env_steps_tl'] = 100
    params['seed'] = 1000
    params['tr_steps_cl'] = 2000000
    # params['eval_epsd_ql'] = 2
    params.pop('tr_epsd_ql', None)
    params.pop('tr_steps_ql', None)
    params.pop('eval_steps_ql', None)
    params.pop('env_steps_ql', None)
    params['eval_epsd_ql'] = 5
    params['env_names_tl'] = [
                                'AntCrossMaze-v3'
                            ]
    params['env_steps_tl'] = 5
    params['tr_epsd_tl'] = 1000
    params['tr_steps_tl'] = 1800
    params['eval_epsd_tl'] = 18
    params.pop('render', None)
    params['eval_epsd_interval'] = 20
    params['masked_done'] = masked_done
    params['active_RND'] = active_RND
    params['active_MC'] = active_MC
    agent_params = pickle.load(open(path+'/agent_params.p','rb'))
    agent_params['dims'] = {
                        'init_prop': 2,
                        'last_prop': 93,
                        'init_ext': 2,
                        'last_ext': 93,
                    }
    agent_params['batch_size']['tl'] = 256
    agent_params['lr']['tl'] = {
                                'target': 5e-3,
                                'beta': 5e-3,
                                'eta': 5e-2
                            }
    agent_params.pop('tl_type', None)
    agent_params.pop('stoA_learning_type', None)
    agent_params['intrinsic_learning'] = False
    # agent_params['init_beta'] = 0.2
    # agent_params['min_beta'] = 1e-2
    # agent_params['delta_beta'] = 5e-5
    # agent_params['alpha']['sl'] /= 4.0
    # agent_params['init_threshold_entropy_alpha'] = -8.0
    # agent_params['classification_with_entropies'] = False
    # agent_params['dim_excluded']['last_2'] = 60
    agent_params['n_concepts'] = 20    
    # agent_params.pop('alpha', None)
    # agent_params.pop('beta', None)
    # agent_params.pop('lr', None)
    # agent_params.pop('init_threshold_entropy_alpha_cl', None)
    # agent_params.pop('min_epsilon_tl', None)
    # agent_params.pop('gamma_tl', None)
    agent_params.pop('memory_capacity', None)
    agent_params.pop('per', None)
    agent_params.pop('clip_value', None)
    agent_params.pop('decision_type', None)
    # agent_params['init_epsilon'] = 0.05
    if initialize_eps:
        agent_params.pop('init_epsilon', None)
    agent_params.pop('min_epsilon', None)
    agent_params.pop('delta_epsilon', None)
    agent_params.pop('init_beta', None)
    agent_params.pop('init_eta', None)
    agent_params.pop('max_concept_divergence', None)
    agent_params['DQL_epsds_target_update'] = 1000
    # agent_params['dim_excluded']['last_2'] = 0
    # agent_params['alpha']['tl'] = 0.05
    # agent_params['alpha']['ql'] = 0.1
    agent_params.pop('gamma_I', None)
    agent_params.pop('gamma_E', None)
    # agent_params['lr']['cl'] = {'alpha': 3e-4}
    print("Params loaded")
    if skill_learning or q_learning:  
        suffix = '_sl' if skill_learning else '_ql'
    if (skill_learning or q_learning) and train:
        try:      
            rewards = list(np.loadtxt(path + '/mean_rewards'+suffix+'.txt'))        
        except:
            rewards = []        

        try:      
            metrics = list(np.loadtxt(path + '/metrics'+suffix+'.txt'))        
        except:        
            metrics = []
    else:
        rewards, metrics = [], []
    
    if concept_learning and train:
        suffix_2 = '_we' if agent_params['classification_with_entropies'] else '_woe' 
        try:      
            losses = list(np.loadtxt(path + '/concept_training_losses'+suffix_2+'.txt')) if last_iter_cl > 0 else []     #TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO   
        except:
            losses = []
            
        try:      
            entropies = list(np.loadtxt(path + '/concept_training_entropies'+suffix_2+'.txt')) if last_iter_cl > 0 else []       
        except:
            entropies = []

        try:      
            entropies_2 = list(np.loadtxt(path + '/concept_training_entropies_2'+suffix_2+'.txt')) if last_iter_cl > 0 else []       
        except:
            entropies_2 = []
    else:
        losses, entropies, entropies_2 = [], [], []
    
    print("Files loaded")
            
    system = System(params, agent_params=agent_params, skill_learning=skill_learning)
    # system.steps['tr']['cl'] = 2000000
    # system.agent.alpha['cl'] = 1e-2
    # system.env_names['tl'] = [
    #                             'AntCrossMaze-v3'
    #                         ]
    # system.n_tasks['tl'] = 1
    # system.agent.threshold_entropy_alpha['cl'] = np.log(10)
    print("System initialized")
    system.load(path, iter_0_sl=last_iter_sl, iter_0_sl_2=last_iter_sl_2, iter_0_ql=last_iter_ql, iter_0_cl=last_iter_cl, iter_0_tl=last_iter_tl, load_memory=load_memory)
    # system.agent.classifier.classifier.optimizer.param_groups[0]['lr'] /= 10.0
    # system.agent.classifier.model_optimizer.param_groups[0]['lr'] /= 10.0
    print("Nets loaded")

    started = True
    initialization = False

    # except:  
    #     print("Error loading")      
    #     started = False
    #     initialization = True

if not started:
    env_names_sl = [
                    # 'HopperLeft-v2',
                    # 'HopperRight-v2'
                    # 'AntStraightLine-v3',
                    # 'AntRotate-v3',
                    # 'AntRotateClock-v3'
                    'AntMT-v3'
                ]

    env_names_ql = [
                    # 'HopperLeft-v2',
                    # 'HopperRight-v2'
                    'AntSquareWall-v3',
                    'AntSquareTrack-v3',
                    'AntGather-v3', 
                    'AntAvoid-v3'
                ]

    params = {
                'env_names_sl': env_names_sl,
                'env_names_ql': env_names_ql                        
            }

    agent_params = {
                    # 'dim_excluded': {
                    #                     'init': 2,
                    #                     'middle': 2,
                    #                     'last': 0,
                    #                     'last_2': 60
                    #                 }
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

last_iter = last_iter_sl if skill_learning else (last_iter_ql if q_learning else (last_iter_cl if concept_learning else last_iter_tl))
if train:
    system.train_agent(initialization=initialization, skill_learning=skill_learning, storing_path=path, 
                        rewards=rewards, metrics=metrics, losses=losses, entropies=entropies,
                        iter_0=last_iter, q_learning=q_learning, concept_learning=concept_learning,
                        transfer_learning=transfer_learning)
if classify:
    if load_data:
        data = np.loadtxt(path + '/eval_events_'+str(task)+'.txt')
        system.agent.classify(T=task, path=path+"/", data=data)
    else:
        system.agent.classify(T=task, path=path+"/")

if eval_agent:
    lt = 'ql' if q_learning else 'sl'
    system.learning_type = lt
    system.set_envs()
    rewards, events, metric_vector, epsd_lenghts = system.eval_agent_skills(eval_epsds=100, explore=False, iter_=0, start_render=True, print_space=True, specific_path='video', max_step=0, task=task)
    np.savetxt(path + '/eval_rewards_'+str(task)+'.txt', np.array(rewards))
    np.savetxt(path + '/eval_events_'+str(task)+'.txt', np.array(events))
    np.savetxt(path + '/eval_metrics_'+str(task)+'.txt', np.array(metric_vector))
    np.savetxt(path + '/eval_lenghts_'+str(task)+'.txt', np.array(epsd_lenghts))
