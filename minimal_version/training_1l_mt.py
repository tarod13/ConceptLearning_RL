import numpy as np

import gym
from agent_1l_mt import create_first_level_multitask_agent 
from buffers import ExperienceBuffer
from trainers import First_Level_Trainer as Trainer

from utils import numpy2torch as np2torch

import wandb
import argparse

DEFAULT_ENV_NAME = 'PendulumMT-v0'
DEFAULT_N_STEPS_IN_EPISODE = 200
DEFAULT_BUFFER_SIZE = 500000
DEFAULT_N_EPISODES = 500
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_BATCH_SIZE = 256 
DEFAULT_LR = 3e-4
DEFAULT_INITIALIZATION = True
DEFAULT_INITIAL_BUFFER_SIZE = 10000
DEFAULT_NOISY_ACTOR_CRITIC = False
DEFAULT_ACTIVE_MULTITASK = True
DAFAULT_DC_TORQUE = True

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--eval", action="store_true", help="Train (False) or evaluate (True) the agent")
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--n_steps_in_episode", default=DEFAULT_N_STEPS_IN_EPISODE, help="Number of steps taken in each episode, default=" + 
        str(DEFAULT_N_STEPS_IN_EPISODE))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--n_episodes", default=DEFAULT_N_EPISODES, help="Number of episodes, default=" + str(DEFAULT_N_EPISODES))
    parser.add_argument("--discount_factor", default=DEFAULT_DISCOUNT_FACTOR, help="Discount factor (0,1), default=" + str(DEFAULT_DISCOUNT_FACTOR))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Batch size, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--initialization", default=DEFAULT_INITIALIZATION, help="Initialize the replay buffer of the agent by acting randomly for a specified number of steps ")
    parser.add_argument("--init_buffer_size", default=DEFAULT_INITIAL_BUFFER_SIZE, help="Minimum replay buffer size to start learning, default=" + str(DEFAULT_INITIAL_BUFFER_SIZE))
    parser.add_argument("--load_id", default=None, help="Model ID to load, default=None")
    parser.add_argument("--load_best", action="store_true", help="If flag is used the best model will be loaded (if ID is provided)")
    parser.add_argument("--noisy_ac", default=DEFAULT_NOISY_ACTOR_CRITIC, help="Use noisy layers in the actor-critic module")
    args = parser.parse_args()

    MODEL_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
    project_name = 'multitaskSAC_first_level'
    
    # Set hyperparameters
    env_name = args.env_name
    n_steps_in_episode = args.n_steps_in_episode
    buffer_size = args.buffer_size
    n_episodes = 1 if args.eval else args.n_episodes
    initialization = args.initialization
    init_buffer_size = args.init_buffer_size
    noisy = args.noisy_ac
    optimizer_kwargs = {
        'batch_size': args.batch_size, 
        'discount_factor': args.discount_factor,
        'lr': args.lr,
    }

    store_video = args.eval
    wandb_project = not args.eval

    # Initilize Weights-and-Biases project
    if wandb_project:
        wandb.init(project=project_name)

        # Log hyperparameters in WandB project
        wandb.config.update(args)
        wandb.config.active_multitask = DEFAULT_ACTIVE_MULTITASK
        wandb.config.active_dc_torque = DAFAULT_DC_TORQUE

    env = gym.make(env_name)

    agent = create_first_level_multitask_agent(env, noisy=noisy)
    if args.load_id is not None:
        if args.load_best:
            agent.load(MODEL_PATH + 'best_', args.load_id)
        else:
            agent.load(MODEL_PATH, args.load_id)

    database = ExperienceBuffer(buffer_size, level=1)

    trainer = Trainer(optimizer_kwargs=optimizer_kwargs)
    returns = trainer.loop(env, agent, database, n_episodes=n_episodes, render=args.render, 
                            max_episode_steps=n_steps_in_episode, 
                            store_video=store_video, wandb_project=wandb_project, 
                            MODEL_PATH=MODEL_PATH, train=(not args.eval),
                            initialization=initialization, init_buffer_size=init_buffer_size)
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 