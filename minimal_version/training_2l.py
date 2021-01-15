import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_2l import create_second_level_agent
from buffers import PixelExperienceBuffer
from trainer import Trainer

from utils import numpy2torch as np2torch
from wrappers import AntPixelWrapper

import wandb

if __name__ == "__main__":    
    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    render = True    
    MODEL_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
    project_name = 'visualSAC_second_level'
    
    # Set hyperparameters
    env_name = 'AntSquareWall-v3'
    n_steps_in_second_level_episode = 600
    buffer_size = 20000
    n_episodes = 1500

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
    returns = trainer.loop(env, agent, database, n_episodes=n_episodes, render=False, 
                            max_episode_steps=n_steps_in_second_level_episode, 
                            store_video=False, wandb_project=True, MODEL_PATH=MODEL_PATH)
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 