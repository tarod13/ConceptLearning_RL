import collections
import numpy as np
import gym


class AntPixelWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return AntPixelWrapper.separate_state(obs)

    @staticmethod
    def separate_state(obs):
        state = collections.OrderedDict()
        state['inner_state'] = obs['state'][2:-60] # Eliminate the xy coordinates (first 2 entries) and the 'lidar' maze observations
        outer_state = obs['pixels'].astype(np.float) / 255.0
        outer_state = np.swapaxes(outer_state, 1, 2)
        state['outer_state'] = np.swapaxes(outer_state, 0, 1)
        state['first_level_obs'] = np.concatenate((obs['state'][2:-61],obs['state'][-60:]))
        return state