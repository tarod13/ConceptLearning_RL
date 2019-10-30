import numpy as np
import torch
from DRIM import System
import pickle

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

valid_agents = ['Hopper', 'Ant']

agent = 'Ant'

if agent == 'Hopper':

    n_test = 126
    iter_ = 200
    iters_posterior = [150,175,190,195,200,205,210,225,250,275,300,400]
    eval_epsds = 1

    env_names = ['Hopper-v2']
    folder_name = 'HopperTest'

elif agent == 'Ant':

    n_test = 101
    iter_ = 600
    iters_posterior = list(np.linspace(25,600,24, dtype='int'))
    eval_epsds = 4

    env_names = [ 'AntLeft-v3', 'AntRight-v3' ]
    folder_name = 'AntLeftRight_v2'

else:
    pass

if agent in valid_agents:
    common_path = folder_name + '/' + str(n_test)
    specific_path = common_path + '/' + str(iter_)

    params = pickle.load(open(common_path+'/params.p','rb'))
    agent_params = pickle.load(open(common_path+'/agent_params.p','rb'))
    print("Params loaded")        
    system = System(params, agent_params=agent_params)
    print("System initialized")
    system.load(common_path, specific_path, load_memory=False)
    print("Nets loaded")

    system.store_video = True
    rewards, _, events, metrics, epsd_lenghts = system.eval_agent(eval_epsds, iter_=iter_, act_randomly=False, start_render=True, print_space=True, specific_path=specific_path)

    np.savetxt(specific_path + '_eval_metrics.txt', np.array(metrics))
    np.savetxt(specific_path + '_eval_rewards.txt', np.array(rewards))
    np.savetxt(specific_path + '_eval_events.txt', events)
    np.savetxt(specific_path + '_eval_lenghts.txt', epsd_lenghts)

    states = torch.FloatTensor(events[:,:system.s_dim]).to(device)

    for i in iters_posterior:
        new_specific_path = common_path + '/' + str(i)
        system.load(common_path, new_specific_path, load_memory=False)
        print("Nets loaded (" + str(i) + ")")

        S, S_upper_posterior = system.agent.concept_model.sample_m_state_and_posterior(states)[:2]
        S, S_upper_posterior = S.detach().cpu().numpy(), S_upper_posterior.detach().cpu().numpy()

        np.savetxt(specific_path + '_eval_posteriors_' + str(i) + '.txt', S_upper_posterior)
        np.savetxt(specific_path + '_eval_mstates_' + str(i) + '.txt', S)