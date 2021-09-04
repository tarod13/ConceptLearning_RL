import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_3l import create_third_level_agent
from buffers import ExperienceBuffer
from trainers import Third_Level_Trainer as Trainer

from utils import numpy2torch as np2torch
from wrappers import AntPixelWrapper

import wandb
import argparse
import os
import collections


DEFAULT_BUFFER_SIZE = 200000
DEFAULT_BATCH_SIZE = 64 
DEFAULT_BATCH_SIZE_C = 256 
DEFAULT_CLIP_VALUE = 1.0
DEFAULT_CLIP_RATIOS = False
DEFAULT_COLLISION_DETECTION = False
DEFAULT_CONCEPT_MODEL_ID = None #'2021-01-30_00-30-28'
DEFAULT_CONTACT_COST = 0.0
DEFAULT_CONTROL_COST = 2e-1
DEFAULT_DEAD_COST = 0.0
DEFAULT_DELTA_EPSILON = 1e-5
DEFAULT_DELTA_EPSILON_MC = (1e-4)*600./8.
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_DISTRIBUTED_CONTRIBUTION = True
DEFAULT_ENTROPY_FACTOR = 0.05
DEFAULT_ENTROPY_UPDATE_RATE = 0.005
DEFAULT_ENV_NAME = 'AntSquareWall-v3'
DEFAULT_EVAL_MC = False
DEFAULT_FORGETTING_FACTOR = 0.0
DEFAULT_HEALTHY_REWARD = 1.0e-1
DEFAULT_ID = None #'2001-01-15_19-10-56'
DEFAULT_INIT_EPSILON = 0.05
DEFAULT_INIT_EPSILON_MC = 0.05
DEFAULT_INIT_LOG_ALPHA = 1.0
DEFAULT_MARGINAL_UPDATE_RATE = 0.05
DEFAULT_MC_ENTROPY = False
DEFAULT_MC_UPDATE_RATE = 1e-1
DEFAULT_MIN_EPSILON = 1.0
DEFAULT_LR = 1e-4
DEFAULT_LR_ALPHA = 1e-4
DEFAULT_LR_ACTOR = 1e-4
DEFAULT_LR_C = 1e-4
DEFAULT_LR_C_ALPHA = (1.e-3)*600./8.
DEFAULT_ALPHA_V_WEIGHT = 0
DEFAULT_INITIALIZATION = False
DEFAULT_INITIAL_BUFFER_SIZE = 500
DEFAULT_N_AGENTS = 1
DEFAULT_N_CONCEPTS = 20
DEFAULT_N_EPISODES = 3000
DEFAULT_N_HEADS = 2
DEFAULT_N_STEP_TD = 1
DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE = 600
DEFAULT_NOISY_ACTOR_CRITIC = False
DEFAULT_PARALLEL_Q_NETS = True
DEFAULT_PRIOR_LOSS_TYPE = 'KL_DIV'
DEFAULT_PRIOR_WEIGHT = 1.0
DEFAULT_REST_N_MC = 0
DEFAULT_SAVE_STEP_EACH = 1
DEFAULT_TRAIN_EACH = 8
DEFAULT_TRAIN_N_MC = 5
DEFAULT_VISION_LATENT_DIM = 64
DEFAULT_WEIGHT_Q_LOSS = 0.5



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_v_weight", default=DEFAULT_ALPHA_V_WEIGHT, help="Weight for entropy velocity in temperature loss, default=" + str(DEFAULT_ALPHA_V_WEIGHT))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Batch size, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--batch_size_c", default=DEFAULT_BATCH_SIZE_C, help="Batch size for high level, default=" + str(DEFAULT_BATCH_SIZE_C))
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--eval", action="store_true", help="Train (False) or evaluate (True) the agent")
    parser.add_argument("--eval_MC", default=DEFAULT_EVAL_MC, help="Evaluate MC policy or SAC actor")
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--n_steps_in_second_level_episode", default=DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE, help="Number of second decision" +
        "level steps taken in each episode, default=" + str(DEFAULT_N_STEPS_IN_SECOND_LEVEL_EPISODE))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--n_episodes", default=DEFAULT_N_EPISODES, help="Number of episodes, default=" + str(DEFAULT_N_EPISODES))
    parser.add_argument("--discount_factor", default=DEFAULT_DISCOUNT_FACTOR, help="Discount factor (0,1), default=" + str(DEFAULT_DISCOUNT_FACTOR))
    parser.add_argument("--init_epsilon", default=DEFAULT_INIT_EPSILON, help="Initial annealing factor for entropy, default=" + str(DEFAULT_INIT_EPSILON))
    parser.add_argument("--init_epsilon_MC", default=DEFAULT_INIT_EPSILON_MC, help="Initial annealing factor for MC entropy, default=" + str(DEFAULT_INIT_EPSILON_MC))
    parser.add_argument("--min_epsilon", default=DEFAULT_MIN_EPSILON, help="Minimum annealing factor for entropy, default=" + str(DEFAULT_MIN_EPSILON))
    parser.add_argument("--delta_epsilon", default=DEFAULT_DELTA_EPSILON, help="Decreasing rate of annealing factor for entropy, default=" + str(DEFAULT_DELTA_EPSILON))
    parser.add_argument("--delta_epsilon_MC", default=DEFAULT_DELTA_EPSILON_MC, help="Decreasing rate of annealing factor for MC entropy, default=" + str(DEFAULT_DELTA_EPSILON_MC))
    parser.add_argument("--entropy_factor", default=DEFAULT_ENTROPY_FACTOR, help="Entropy coefficient, default=" + str(DEFAULT_ENTROPY_FACTOR))
    parser.add_argument("--MC_entropy", default=DEFAULT_MC_ENTROPY, help="Add entropy to Monte Carlo learning, default=" + str(DEFAULT_MC_ENTROPY))
    parser.add_argument("--MC_update_rate", default=DEFAULT_MC_UPDATE_RATE, help="Monte Carlo learning update rate, default=" + str(DEFAULT_MC_UPDATE_RATE))
    parser.add_argument("--entropy_update_rate", default=DEFAULT_ENTROPY_UPDATE_RATE, help="Mean entropy update rate, default=" + str(DEFAULT_ENTROPY_UPDATE_RATE))
    parser.add_argument("--marginal_update_rate", default=DEFAULT_MARGINAL_UPDATE_RATE, help="Marginal concept distribution update rate, default=" + str(DEFAULT_MARGINAL_UPDATE_RATE))
    parser.add_argument("--forgetting_factor", default=DEFAULT_FORGETTING_FACTOR, help="Factor that reduces MC counts (0,1], default=" + str(DEFAULT_FORGETTING_FACTOR))
    parser.add_argument("--weight_q_loss", default=DEFAULT_WEIGHT_Q_LOSS, help="Weight of critics' loss, default=" + str(DEFAULT_WEIGHT_Q_LOSS))
    parser.add_argument("--init_log_alpha", default=DEFAULT_INIT_LOG_ALPHA, help="Initial temperature parameter, default=" + str(DEFAULT_INIT_LOG_ALPHA))
    parser.add_argument("--load_id", default=None, help="Model ID to load, default=None")
    parser.add_argument("--load_concept_id", default=DEFAULT_CONCEPT_MODEL_ID, help="ID of concept model to load, default="+DEFAULT_CONCEPT_MODEL_ID)
    parser.add_argument("--load_best", action="store_true", help="If flag is used the best model will be loaded (if ID is provided)")
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--lr_alpha", default=DEFAULT_LR_ALPHA, help="Learning rate for temperature, default=" + str(DEFAULT_LR_ALPHA))
    parser.add_argument("--lr_actor", default=DEFAULT_LR_ACTOR, help="Learning rate for actor, default=" + str(DEFAULT_LR_ACTOR))
    parser.add_argument("--lr_c", default=DEFAULT_LR_C, help="Learning rate for high level, default=" + str(DEFAULT_LR_C))
    parser.add_argument("--lr_c_Alpha", default=DEFAULT_LR_C_ALPHA, help="Learning rate for temperature in high level, default=" + str(DEFAULT_LR_C_ALPHA))
    parser.add_argument("--prior_weight", default=DEFAULT_PRIOR_WEIGHT, help="Weight for concept prior in actor loss, default=" + str(DEFAULT_PRIOR_WEIGHT))
    parser.add_argument("--prior_loss_type", default=DEFAULT_PRIOR_LOSS_TYPE, help="Loss associated with prior, default=" + str(DEFAULT_PRIOR_LOSS_TYPE))
    parser.add_argument("--initialization", default=DEFAULT_INITIALIZATION, help="Initialize the replay buffer of the agent by acting randomly for a specified number of steps ")
    parser.add_argument("--init_buffer_size", default=DEFAULT_INITIAL_BUFFER_SIZE, help="Minimum replay buffer size to start learning, default=" + str(DEFAULT_INITIAL_BUFFER_SIZE))
    parser.add_argument("--noisy_ac", default=DEFAULT_NOISY_ACTOR_CRITIC, help="Use noisy layers in the actor-critic module")
    parser.add_argument("--save_step_each", default=DEFAULT_SAVE_STEP_EACH, help="Number of steps to store 1 step in the replay buffer, default=" + str(DEFAULT_SAVE_STEP_EACH))
    parser.add_argument("--train_each", default=DEFAULT_TRAIN_EACH, help="Number of steps ellapsed to train once, default=" + str(DEFAULT_TRAIN_EACH))
    parser.add_argument("--train_n_mc", default=DEFAULT_TRAIN_N_MC, help="Number of episodes that MC estimation is performed without learning, default=" + str(DEFAULT_TRAIN_N_MC))
    parser.add_argument("--rest_n_mc", default=DEFAULT_REST_N_MC, help="Number of episodes that MC estimation is not performed, default=" + str(DEFAULT_REST_N_MC))
    parser.add_argument("--n_step_td", default=DEFAULT_N_STEP_TD, help="Number of steps to calculate temporal differences, default=" + str(DEFAULT_N_STEP_TD))
    parser.add_argument("--parallel_q_nets", default=DEFAULT_PARALLEL_Q_NETS, help="Use or not parallel q nets in actor critic, default=" + str(DEFAULT_PARALLEL_Q_NETS))
    parser.add_argument("--n_heads", default=DEFAULT_N_HEADS, help="Number of heads in the critic, default=" + str(DEFAULT_N_HEADS))
    parser.add_argument("--clip_value", default=DEFAULT_CLIP_VALUE, help="Clip value for optimizer")
    parser.add_argument("--clip_ratios", default=DEFAULT_CLIP_RATIOS, help="Clip importance sampling ratios")
    parser.add_argument("--distributed_contribution", default=DEFAULT_DISTRIBUTED_CONTRIBUTION, help="Distribute state contrubtion to all concepts")
    parser.add_argument("--n_agents", default=DEFAULT_N_AGENTS, type=int, help="Number of agents")
    parser.add_argument("--n_concepts", default=DEFAULT_N_CONCEPTS, type=int, help="Number of concepts")
    parser.add_argument("--vision_latent_dim", default=DEFAULT_VISION_LATENT_DIM, help="Dimensionality of feature vector added to inner state, default=" + 
        str(DEFAULT_VISION_LATENT_DIM))
    args = parser.parse_args()

    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    MODEL_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'
    concept_path = MODEL_PATH + 'concept_models/'
    project_name = 'visualSAC_second_level'
    
    # Set hyperparameters
    env_name = args.env_name
    n_steps_in_second_level_episode = args.n_steps_in_second_level_episode
    buffer_size = args.buffer_size
    n_episodes = 1 if args.eval else args.n_episodes
    initialization = args.initialization
    init_buffer_size = args.init_buffer_size
    noisy = args.noisy_ac
    save_step_each = args.save_step_each
    n_step_td = args.n_step_td
    n_heads = args.n_heads
    optimizer_kwargs = {
        'batch_size': args.batch_size, 
        'discount_factor': args.discount_factor,
        'init_epsilon': args.init_epsilon,
        'min_epsilon': args.min_epsilon,
        'delta_epsilon': args.delta_epsilon,
        'entropy_factor': args.entropy_factor,
        'weight_q_loss': args.weight_q_loss,
        'alpha_v_weight': args.alpha_v_weight,
        'entropy_update_rate': args.entropy_update_rate,
        'clip_value': args.clip_value,
        'marginal_update_rate': args.marginal_update_rate,
        'prior_weight': args.prior_weight,
        'batch_size_c': args.batch_size_c,
        'prior_loss_type': args.prior_loss_type,
        'clip_ratios': args.clip_ratios,
        'distributed_contribution': args.distributed_contribution,
        'MC_entropy': args.MC_entropy,
        'MC_update_rate': args.MC_update_rate,
        'forgetting_factor': args.forgetting_factor,
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
    
    
    agent = create_third_level_agent(concept_path, args.load_concept_id, args.n_concepts, noisy=noisy, 
        n_heads=n_heads, init_log_alpha=args.init_log_alpha, latent_dim=args.vision_latent_dim, 
        parallel=args.parallel_q_nets, lr=args.lr, lr_alpha=args.lr_alpha, lr_actor=args.lr_actor, min_entropy_factor=args.entropy_factor, 
        lr_c=args.lr_c, lr_Alpha=args.lr_c_Alpha, entropy_update_rate=args.entropy_update_rate, init_Epsilon=args.init_epsilon_MC,
        delta_Epsilon=args.delta_epsilon_MC)
    
    if args.load_id is not None:
        if args.load_best:
            agent.load(MODEL_PATH + env_name + '/best_', args.load_id)
        else:
            agent.load(MODEL_PATH + env_name + '/last_', args.load_id)
    agents = collections.deque(maxlen=args.n_agents)
    agents.append(agent)
    
    os.makedirs(MODEL_PATH + env_name, exist_ok=True)

    database = ExperienceBuffer(buffer_size, level=2)

    trainer = Trainer(optimizer_kwargs=optimizer_kwargs)
    returns = trainer.loop(env, agents, database, n_episodes=n_episodes, render=args.render, 
                            max_episode_steps=n_steps_in_second_level_episode, 
                            store_video=store_video, wandb_project=wandb_project, 
                            MODEL_PATH=MODEL_PATH, train=(not args.eval),
                            initialization=initialization, init_buffer_size=init_buffer_size,
                            save_step_each=save_step_each, train_each=args.train_each, 
                            n_step_td=n_step_td, train_n_MC=args.train_n_mc, rest_n_MC=args.rest_n_mc,
                            eval_MC=args.eval_MC)
    G = returns.mean()    
    print("Mean episode return: {:.2f}".format(G)) 