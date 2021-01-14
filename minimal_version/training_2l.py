import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_2l import create_second_level_agent
from buffers import PixelExperienceBuffer
from trainer import Trainer

from utils import numpy2torch as np2torch
from utils import AntPixelWrapper


if __name__ == "__main__":
    env_name = 'AntAvoid-v3'
    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    render = True
    n_steps = 1000
    
    env = AntPixelWrapper( 
            PixelObservationWrapper(gym.make(env_name).unwrapped,
                                    pixels_only=False,
                                    render_kwargs=render_kwargs.copy())
    )
    agent = create_second_level_agent()
    database = PixelExperienceBuffer(2000, level=2)

    trainer = Trainer(train_level=2)
    returns = trainer.loop(env, agent, database, n_episodes=1, render=False, 
                            max_episode_steps=n_steps, store_video=True)
    G = returns.mean()

    print("Mean episode return: {:.2f}".format(G))