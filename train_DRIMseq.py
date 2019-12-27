import numpy as np
from DRIMseq import System
import os

def exists_folder(f_name):
    return os.path.isdir(f_name)

env_names_sl = [
                    'AntStraightLine-v3',
                    'AntRotate-v3',
                    'AntRotateClock-v3'
                ]

env_names_cl = [
                    'AntSquareTrack-v3',
                    'AntGatherRewards-v3', 
                    'AntGatherBombs-v3'
                ]

params = {
                'seed': 1000,
                'env_names_sl': env_names_sl,
                'env_names_cl': env_names_cl,
                'env_names_tl': [],
                'env_steps': 1, 
                'grad_steps': 1, 
                'init_steps': 1000,
                'max_episode_steps': 1000,
                'batch_size': 256, 
                'render': True, 
                'reset_when_done': True, 
                'store_video': False                            
            }

system = System(params)


