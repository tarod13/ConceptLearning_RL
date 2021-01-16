import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_2l import create_second_level_agent
from buffers import PixelExperienceBuffer
from trainer import Trainer

from utils import numpy2torch as np2torch
from wrappers import AntPixelWrapper

import wandb
import argparse

DEFAULT_ENV_NAME = 'AntSquareWall-v3'
DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE = 600
DEFAULT_BUFFER_SIZE = 20000
DEFAULT_N_EPISODES = 1500

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", default=False, action="store_true", help="Disable cuda")
    parser.add_argument("--render", default=False, action="store_true", help="Display agent-env interaction")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--nsteps", default=DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE, help="Number of second decision" +
        "level steps taken in each episode, default=" + DEFAULT_METHOD)
    parser.add_argument("--bsize", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + DEFAULT_BUFFER_SIZE)
    parser.add_argument("--nepsds", default=DEFAULT_N_EPISODES, help="Number of episodes, default=" + DEFAULT_N_EPISODES)
    args = parser.parse_args()

    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    MODEL_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
    project_name = 'visualSAC_second_level'
    
    # Set hyperparameters
    env_name = args.env
    n_steps_in_second_level_episode = args.nsteps
    buffer_size = args.bsize
    n_episodes = args.nepsds

    # Initilize Weights-and-Biases project
    wandb.init(project=project_name)

    # Log hyperparameters in WandB project
    wandb.config.env_name = env_name
    wandb.config.n_steps_in_second_level_episode = n_steps_in_second_level_episode
    wandb.config.buffer_size = buffer_size
    wandb.config.n_episodes = n_episodes 

    env = AntPixelWrapper( 
            PixelObservationWrapper(gym.make(env_name).unwrapped,
                                    pixels_only=False,
                                    render_kwargs=render_kwargs.copy())
    )
    agent = create_second_level_agent()
    database = PixelExperienceBuffer(buffer_size, level=2)

    trainer = Trainer(train_level=2)
    returns = trainer.loop(env, agent, database, n_episodes=n_episodes, render=args.render, 
                            max_episode_steps=n_steps_in_second_level_episode, 
                            store_video=False, wandb_project=True, MODEL_PATH=MODEL_PATH)
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 