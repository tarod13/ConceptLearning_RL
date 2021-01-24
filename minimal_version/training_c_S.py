import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_c_S import Conceptual_Agent
from concept_optimizers import S_ConceptOptimizer
from buffers import ExperienceBuffer
from trainers import Second_Level_Trainer as Trainer
from wrappers import AntPixelWrapper

import argparse
import pickle
import os
import wandb


LOAD_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_data/'
SAVE_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'


DEFAULT_N_STEPS = 1000
DEFAULT_BUFFER_SIZE = 24000
DEFAULT_BATCH_SIZE = 64
DEFAULT_INNER_STATE_DIM = 32
DEFAULT_VISION_LATENT_DIM = 512
DEFAULT_NOISY = False
DEFAULT_N_PARTS = 20
DEFAULT_DB_ID = '2021-01-23_18-58-31'
DEFAULT_N_TASKS = 4
DEFAULT_N_CONCEPTS = 20
DEFAULT_N_ACTIONS = 4
DEFAULT_LR = 3e-4


def load_database(n_parts, DB_ID, buffer_size ,level):
    database = ExperienceBuffer(buffer_size, level)
    for i in range(0, n_parts):
        PATH = LOAD_PATH + DB_ID + '/SAC_training_level2_database_part_' + str(i) + '.p'
        database.buffer += pickle.load(open(PATH, 'rb'))
    return database
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--n_steps", default=DEFAULT_N_STEPS, help="Number of SGD steps taken, default=" + str(DEFAULT_N_STEPS))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Size of batch used in SGD, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--n_parts", default=DEFAULT_N_PARTS, help="Number of parts in which the database is divided and store, default=" + str(DEFAULT_N_PARTS))
    parser.add_argument("--noisy", default=DEFAULT_NOISY, help="Use noisy layers in the concept module")
    parser.add_argument("--db_id", default=DEFAULT_DB_ID, help="Database ID")
    parser.add_argument("--iinner_state_dim", default=DEFAULT_INNER_STATE_DIM, help="Dimensionality of inner state, default=" + str(DEFAULT_INNER_STATE_DIM))
    parser.add_argument("--vision_latent_dim", default=DEFAULT_VISION_LATENT_DIM, help="Dimensionality of feature vector added to inner state, default=" + 
        str(DEFAULT_VISION_LATENT_DIM))
    parser.add_argument("--n_tasks", default=DEFAULT_N_TASKS, help="Number of tasks, default=" + str(DEFAULT_N_TASKS))
    parser.add_argument("--n_concepts", default=DEFAULT_N_CONCEPTS, help="Number of concepts, default=" + str(DEFAULT_N_CONCEPTS))
    parser.add_argument("--n_actions", default=DEFAULT_N_ACTIONS, help="Number of actions, default=" + str(DEFAULT_N_ACTIONS))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    args = parser.parse_args()

    render_kwargs = {'pixels': {'width':168,
                            'height':84,
                            'camera_name':'front_camera'}}
    project_name = 'visualSAC_conceptual_level'

    # Initilize Weights-and-Biases project
    wandb.init(project=project_name)

    # Log hyperparameters in WandB project
    wandb.config.update(args)
    
    database = load_database(args.n_parts, args.db_id, args.buffer_size, 3)
    conceptual_agent = Conceptual_Agent(args.inner_state_dim, args.vision_latent_dim, args.n_concepts, args.noisy)
    concept_optimizer = S_ConceptOptimizer(args.batch_size, lr=args.lr)

    for step in range(0, args.n_steps):
        metrics = concept_optimizer.optimize(conceptual_agent, database, args.n_actions, args.n_tasks)
        metrics['step'] = step
        wandb.log(metrics)
