import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_2l import create_second_level_agent
from buffers import ExperienceBuffer, Task, PixelExperienceSecondLevelMT
from trainers import Second_Level_Trainer as Trainer
from wrappers import AntPixelWrapper
from utils import load_env_model_pairs

import argparse
import pickle
from utils import time_stamp
import itertools
import os


LOAD_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
SAVE_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_data/'

DEFAULT_FILE = 'models_to_load_l2.yaml'
DEFAULT_N_STEPS = 600
DEFAULT_BUFFER_SIZE = 100000
DEFAULT_SAVE_STEP_EACH = 1
DEFAULT_USE_SAC_BASELINES = False
DEFAULT_N_HEADS = 2
DEFAULT_VISION_LATENT_DIM = 64
DEFAULT_NOISY_ACTOR_CRITIC = False
DEFAULT_N_PARTS = 20
DEFAULT_PARALLEL_Q_NETS = True



def load_agent(env_name, model_id, load_best=True, sac_baselines=False, 
                noisy=True, n_heads=2, latent_dim=256, parallel=True):
    agent = create_second_level_agent(noisy=noisy, n_heads=n_heads, 
                            latent_dim=latent_dim, parallel=parallel)        
    if load_best:
        agent.load(LOAD_PATH + '/' + env_name + '/best_', model_id)
    else:
        agent.load(LOAD_PATH + '/' + env_name + '/last_', model_id)
    return agent


def store_database(database, n_parts):
    part_size = len(database.buffer) // n_parts
    DB_ID = time_stamp()

    os.makedirs(SAVE_PATH + DB_ID)

    for i in range(0, n_parts):
        PATH = SAVE_PATH + DB_ID + '/SAC_training_level2_database_part_' + str(i) + '.p'

        if (i+1) < n_parts:
            pickle.dump(list(itertools.islice(database.buffer, part_size*i, part_size*(i+1))), open(PATH, 'wb'))
        else:
            pickle.dump(list(itertools.islice(database.buffer, part_size*i, None)), open(PATH, 'wb'))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Name of the folder wih the model info. needed to load them, default=" + DEFAULT_FILE)
    parser.add_argument("--n_steps", default=DEFAULT_N_STEPS, help="Number of steps taken in each episode, default=" + str(DEFAULT_N_STEPS))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--n_parts", default=DEFAULT_N_PARTS, help="Number of parts in which the database is divided and store, default=" + str(DEFAULT_N_PARTS))
    parser.add_argument("--load_best", action="store_false", help="If flag is used, the last, instead of the best, model will be loaded")
    parser.add_argument("--noisy_ac", default=DEFAULT_NOISY_ACTOR_CRITIC, help="Use noisy layers in the actor-critic module")
    parser.add_argument("--save_step_each", default=DEFAULT_SAVE_STEP_EACH, help="Number of steps to store 1 step in the replay buffer, default=" + str(DEFAULT_SAVE_STEP_EACH))
    parser.add_argument("--sac_baselines", default=DEFAULT_USE_SAC_BASELINES, help="Use SAC baselines if flag is given")
    parser.add_argument("--parallel_q_nets", default=DEFAULT_PARALLEL_Q_NETS, help="Use or not parallel q nets in actor critic, default=" + str(DEFAULT_PARALLEL_Q_NETS))
    parser.add_argument("--n_heads", default=DEFAULT_N_HEADS, help="Number of heads in the critic, default=" + str(DEFAULT_N_HEADS))
    parser.add_argument("--vision_latent_dim", default=DEFAULT_VISION_LATENT_DIM, help="Dimensionality of feature vector added to inner state, default=" + 
        str(DEFAULT_VISION_LATENT_DIM))
    args = parser.parse_args()

    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    
    
    database = ExperienceBuffer(args.buffer_size, level=3)
    trainer = Trainer()

    env_model_pairs = load_env_model_pairs(args.file)
    n_envs = len(env_model_pairs)
    n_episodes = (args.buffer_size * args.save_step_each) // args.n_steps
    store_video = False

    for env_number, (env_name, model_id) in enumerate(env_model_pairs.items()):
        task_database = ExperienceBuffer(args.buffer_size//n_envs, level=2)

        env = AntPixelWrapper( 
                PixelObservationWrapper(gym.make(env_name).unwrapped,
                                        pixels_only=False,
                                        render_kwargs=render_kwargs.copy())
        )

        agent = load_agent(env_name, model_id, args.load_best, args.sac_baselines, 
                            args.noisy_ac, args.n_heads, args.vision_latent_dim,
                            args.parallel_q_nets)
    
        returns = trainer.loop(env, [agent], task_database, n_episodes=n_episodes, render=args.render, 
                                max_episode_steps=args.n_steps, store_video=False, wandb_project=False, 
                                MODEL_PATH=LOAD_PATH, train=False, initialization=False,
                                save_step_each=args.save_step_each, n_step_td=1)
        G = returns.mean()    
        print("Env: " + env_name + ", Mean episode return: {:.2f}".format(G))

        for experience in task_database.buffer:
            task = Task(task=env_number)
            experience_with_task_info = PixelExperienceSecondLevelMT(*experience, *task)
            database.append(experience_with_task_info) 
    
    store_database(database, args.n_parts)
    print("Database stored succesfully")
