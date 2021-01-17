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
DEFAULT_BUFFER_SIZE = 48000
DEFAULT_N_EPISODES = 1500
DEFAULT_ID = '2001-01-15_19-10-56'
DEFAULT_CONTROL_COST = 5e-1
DEFAULT_COLLISION_DETECTION = False
DEFAULT_CONTACT_COST = 5e-4
DEFAULT_HEALTHY_REWARD = 1.0
DEFAULT_DEAD_COST = 0.0
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_BATCH_SIZE = 64 
DEFAULT_MIN_EPSILON = np.log(2)
DEFAULT_INIT_EPSILON = 0.7785
DEFAULT_DELTA_EPSILON = 5e-7
DEFAULT_ENTROPY_FACTOR = 0.95
DEFAULT_WEIGHT_Q_LOSS = 0.5
DEFAULT_WEIGHT_ALPHA_LOSS = 10.0
DEFAULT_LR = 3e-4
DEFAULT_INITIALIZATION = False
DEFAULT_INITIAL_BUFFER_SIZE = 500

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--eval", action="store_true", help="Train (False) or evaluate (True) the agent")
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--n_steps_in_second_level_episode", default=DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE, help="Number of second decision" +
        "level steps taken in each episode, default=" + str(DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--n_episodes", default=DEFAULT_N_EPISODES, help="Number of episodes, default=" + str(DEFAULT_N_EPISODES))
    parser.add_argument("--discount_factor", default=DEFAULT_DISCOUNT_FACTOR, help="Discount factor (0,1), default=" + str(DEFAULT_DISCOUNT_FACTOR))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Batch size, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--init_epsilon", default=DEFAULT_INIT_EPSILON, help="Initial annealing factor for entropy, default=" + str(DEFAULT_INIT_EPSILON))
    parser.add_argument("--min_epsilon", default=DEFAULT_MIN_EPSILON, help="Minimum annealing factor for entropy, default=" + str(DEFAULT_MIN_EPSILON))
    parser.add_argument("--delta_epsilon", default=DEFAULT_DELTA_EPSILON, help="Decreasing rate of annealing factor for entropy, default=" + str(DEFAULT_DELTA_EPSILON))
    parser.add_argument("--entropy_factor", default=DEFAULT_ENTROPY_FACTOR, help="Entropy coefficient, default=" + str(DEFAULT_ENTROPY_FACTOR))
    parser.add_argument("--weight_q_loss", default=DEFAULT_WEIGHT_Q_LOSS, help="Weight of critics' loss, default=" + str(DEFAULT_WEIGHT_Q_LOSS))
    parser.add_argument("--weight_alpha_loss", default=DEFAULT_WEIGHT_ALPHA_LOSS, help="Weight of temperature loss, default=" + str(DEFAULT_WEIGHT_ALPHA_LOSS))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--initialization", default=DEFAULT_INITIALIZATION, help="Initialize the replay buffer of the agent by acting randomly for a specified number of steps ")
    parser.add_argument("--init_buffer_size", default=DEFAULT_INITIAL_BUFFER_SIZE, help="Minimum replay buffer size to start learning, default=" + str(DEFAULT_INITIAL_BUFFER_SIZE))
    parser.add_argument("--load_id", default=None, help="Model ID to load, default=None")
    args = parser.parse_args()

    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    MODEL_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
    project_name = 'visualSAC_second_level'
    
    # Set hyperparameters
    env_name = args.env_name
    n_steps_in_second_level_episode = args.n_steps_in_second_level_episode
    buffer_size = args.buffer_size
    n_episodes = 1 if args.eval else args.n_episodes
    initialization = args.initialization
    init_buffer_size = args.init_buffer_size
    optimizer_kwargs = {
        'batch_size': args.batch_size, 
        'discount_factor': args.discount_factor,
        'init_epsilon': args.init_epsilon,
        'min_epsilon': args.min_epsilon,
        'delta_epsilon': args.delta_epsilon,
        'entropy_factor': args.entropy_factor,
        'weight_q_loss': args.weight_q_loss,
        'weight_alpha_loss': args.weight_alpha_loss,
        'lr': args.lr,
    }

    store_video = args.eval
    wandb_project = not args.eval

    # Initilize Weights-and-Biases project
    if wandb_project:
        wandb.init(project=project_name)

        # Log hyperparameters in WandB project
        wandb.config.update(args)
        wandb.config.control_cost = DEFAULT_CONTROL_COST
        wandb.config.collision_detect = DEFAULT_COLLISION_DETECTION
        wandb.config.contact_cost = DEFAULT_CONTACT_COST
        wandb.config.dead_cost = DEFAULT_DEAD_COST
        wandb.config.healthy_reward = DEFAULT_HEALTHY_REWARD 


    env = AntPixelWrapper( 
            PixelObservationWrapper(gym.make(env_name).unwrapped,
                                    pixels_only=False,
                                    render_kwargs=render_kwargs.copy())
    )
    agent = create_second_level_agent()
    if args.load_id is not None:
        agent.load(MODEL_PATH, args.load_id)

    database = PixelExperienceBuffer(buffer_size, level=2)

    trainer = Trainer(train_level=2, optimizer_kwargs=optimizer_kwargs)
    returns = trainer.loop(env, agent, database, n_episodes=n_episodes, render=args.render, 
                            max_episode_steps=n_steps_in_second_level_episode, 
                            store_video=store_video, wandb_project=wandb_project, 
                            MODEL_PATH=MODEL_PATH, train=(not args.eval),
                            initialization=initialization, init_buffer_size=init_buffer_size)
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 