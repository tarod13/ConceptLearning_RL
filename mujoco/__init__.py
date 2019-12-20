from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv_old
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv
# from gym.envs.mujoco.ant_cross_maze import AntCrossMazeEnv
import gym.envs.mujoco.utils
#Custom
from gym.envs.mujoco.ant_random_direction import AntRandDirEnv
from gym.envs.mujoco.ant_straight_line import AntStraightLineEnv
from gym.envs.mujoco.ant_straight_line_reversed import AntStraightLineRevEnv
from gym.envs.mujoco.ant_rotate import AntRotateEnv
from gym.envs.mujoco.ant_rotate_reward import AntRotateRewardEnv
from gym.envs.mujoco.ant_rotate_clockwise import AntRotateClockEnv
from gym.envs.mujoco.ant_rotate_anticlockwise import AntRotateAntiClockEnv
from gym.envs.mujoco.ant_gather import AntGatherEnv
from gym.envs.mujoco.ant_gather_rewards import AntGatherREnv
from gym.envs.mujoco.ant_gather_bombs import AntGatherBEnv
from gym.envs.mujoco.ant_el_maze import AntLMazeEnv
from gym.envs.mujoco.ant_square_track import AntSquareTrackEnv
from gym.envs.mujoco.ant_square_track_bomb import AntSquareTrackBombEnv
from gym.envs.mujoco.ant_square_track_reward import AntSquareTrackRewardEnv
from gym.envs.mujoco.ant_square_track_clockwise import AntSquareTrackClockEnv
from gym.envs.mujoco.ant_square_track_anticlockwise import AntSquareTrackAntiClockEnv
from gym.envs.mujoco.ant_left import AntLeftEnv
from gym.envs.mujoco.ant_right import AntRightEnv
from gym.envs.mujoco.ant_v2_cross_maze import AntCrossMazeEnv

