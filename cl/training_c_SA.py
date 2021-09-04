import numpy as np

import gym
from agent_c_SA import create_conceptual_agent
from concept_optimizers import SA_ConceptOptimizer
from buffers import ExperienceBuffer

import argparse
import pickle
import os
import wandb


LOAD_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_data/'
SAVE_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/'


DEFAULT_ENV_NAME = 'PendulumMT-v0'
DEFAULT_N_STEPS = 10000
DEFAULT_BUFFER_SIZE = 100000
DEFAULT_BATCH_SIZE = 128
DEFAULT_NOISY = False
DEFAULT_N_PARTS = 20
DEFAULT_DB_ID = '2021-01-24_20-55-15'
DEFAULT_MODEL_ID = '2021-01-24_11-56-00'
DEFAULT_N_CONCEPTS = 8
DEFAULT_N_ACTIONS = 4
DEFAULT_LR = 3e-6
DEFAULT_INIT_LOG_ALPHA = 0


def load_database(n_parts, DB_ID, buffer_size ,level):
    database = ExperienceBuffer(buffer_size, level)
    for i in range(0, n_parts):
        PATH = LOAD_PATH + DB_ID + '/SAC_training_level1_database_part_' + str(i) + '.p'
        database.buffer += pickle.load(open(PATH, 'rb'))
    return database
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--render", action="store_true", help="Display agent-env interaction")
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--n_steps", default=DEFAULT_N_STEPS, help="Number of SGD steps taken, default=" + str(DEFAULT_N_STEPS))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Size of batch used in SGD, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--n_parts", default=DEFAULT_N_PARTS, help="Number of parts in which the database is divided and store, default=" + str(DEFAULT_N_PARTS))
    parser.add_argument("--noisy", default=DEFAULT_NOISY, help="Use noisy layers in the concept module")
    parser.add_argument("--db_id", default=DEFAULT_DB_ID, help="Database ID")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Multitask actor critic ID")
    parser.add_argument("--n_concepts", default=DEFAULT_N_CONCEPTS, help="Number of concepts, default=" + str(DEFAULT_N_CONCEPTS))
    parser.add_argument("--n_actions", default=DEFAULT_N_ACTIONS, help="Number of actions, default=" + str(DEFAULT_N_ACTIONS))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--init_log_alpha", default=DEFAULT_INIT_LOG_ALPHA, help="Initial temperature parameter, default=" + str(DEFAULT_INIT_LOG_ALPHA))
    args = parser.parse_args()

    project_name = 'multitaskSAC_conceptual_level'

    # Initilize Weights-and-Biases project
    wandb.init(project=project_name)

    # Log hyperparameters in WandB project
    wandb.config.update(args)
    
    env = gym.make(args.env_name)
    database = load_database(args.n_parts, args.db_id, args.buffer_size, 1)
    n_concepts = {'state': args.n_concepts, 'action': args.n_actions}
    conceptual_agent = create_conceptual_agent(env, args.model_id, 
        init_log_alpha=args.init_log_alpha)
    concept_optimizer = SA_ConceptOptimizer(args.batch_size, lr=args.lr)

    for step in range(0, args.n_steps):
        metrics = concept_optimizer.optimize(conceptual_agent, database)
        metrics['step'] = step
        wandb.log(metrics)
