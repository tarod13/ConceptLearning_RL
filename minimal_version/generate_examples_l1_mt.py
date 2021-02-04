import numpy as np

import gym
from agent_1l_mt import create_first_level_multitask_agent
from buffers import ExperienceBuffer, Task, PixelExperienceSecondLevelMT
from trainers import First_Level_Trainer as Trainer
from utils import load_env_model_pairs


import argparse
import pickle
from utils import time_stamp
import itertools
import os


LOAD_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
SAVE_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_data/'

DEFAULT_FILE = 'models_to_load_l1_mt.yaml'
DEFAULT_N_STEPS = 200
DEFAULT_BUFFER_SIZE = 100000
DEFAULT_NOISY_ACTOR_CRITIC = False
DEFAULT_N_PARTS = 20
DEFAULT_LR = 3e-4
DEFAULT_LR_ALPHA = 3e-4


def load_agent(env, model_id, load_best=True, actor_critic_kwargs={}, device='cuda'):
    agent = create_first_level_multitask_agent(env, device=device, actor_critic_kwargs=actor_critic_kwargs)       
    if load_best:
        agent.load(LOAD_PATH + '/' + env.spec.id + '/best_', model_id)
    else:
        agent.load(LOAD_PATH + '/' + env.spec.id + '/last_', model_id)
    return agent


def store_database(database, n_parts):
    part_size = len(database.buffer) // n_parts
    DB_ID = time_stamp()

    os.makedirs(SAVE_PATH + DB_ID)

    for i in range(0, n_parts):
        PATH = SAVE_PATH + DB_ID + '/SAC_training_level1_database_part_' + str(i) + '.p'

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
    parser.add_argument("--load_best", action="store_false", help="If flag is used the last, instead of the best, model will be loaded")
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--lr_alpha", default=DEFAULT_LR_ALPHA, help="Learning rate for temperature parameter alpha, default=" + str(DEFAULT_LR_ALPHA))
    parser.add_argument("--noisy", default=DEFAULT_NOISY_ACTOR_CRITIC, help="Use noisy layers in the actor-critic module")
    args = parser.parse_args()

    actor_critic_kwargs = {
        'noisy': args.noisy, 
        'lr': args.lr,
        'lr_alpha': args.lr_alpha,
    }

    database = ExperienceBuffer(args.buffer_size, level=1)
    trainer = Trainer()

    env_model_pairs = load_env_model_pairs(args.file)
    n_envs = len(env_model_pairs)
    n_episodes = args.buffer_size // args.n_steps
    store_video = False

    for env_number, (env_name, model_id) in enumerate(env_model_pairs.items()):
        env = gym.make(env_name)

        agent = load_agent(env, model_id, args.load_best, actor_critic_kwargs)
    
        returns = trainer.loop(env, agent, database, n_episodes, render=args.render, 
                                max_episode_steps=args.n_steps, store_video=False, wandb_project=False, 
                                MODEL_PATH=LOAD_PATH, train=False, initialization=False)
        G = returns.mean()    
        print("Env: " + env_name + ", Mean episode return: {:.2f}".format(G))

    
    store_database(database, args.n_parts)
    print("Database stored succesfully")
