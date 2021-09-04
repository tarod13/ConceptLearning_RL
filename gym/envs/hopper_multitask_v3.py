import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 2,
    'distance': 3.0,
    'lookat': np.array((0.0, 0.0, 1.15)),
    'elevation': -20.0,
}


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hopper.xml',
                 forward_reward_weight=0.5,
		         height_reward_weight=8.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=0.5,
                 terminate_when_unhealthy=True,
		         desired_height=1.5,
                 desired_velocity=1.0,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-np.pi/4, np.pi/4),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())

        self._n_tasks = 4

        self._forward_reward_weight = forward_reward_weight
        self._height_reward_weight = height_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._desired_height = desired_height
        self._desired_velocity = desired_velocity

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(
            np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(
            self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position[:-4], position[-4:], velocity[-4:], velocity[:-4])).ravel()
        return observation

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after, z_position_after = self.sim.data.qpos[0:2]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        height_error = np.abs(self._desired_height - z_position_after)
        velocity_error = np.abs(self._desired_velocity - x_velocity)

        ctrl_cost = self.control_cost(action)
        healthy_reward = self.healthy_reward

        fast_forward_reward = 3.0 * self._forward_reward_weight * x_velocity
        slow_forward_reward = 4.0 * self._forward_reward_weight * np.exp(-0.5*velocity_error**2/0.25**2)
        clipped_forward_reward = self._forward_reward_weight * np.min([self._desired_velocity, x_velocity])
        
        if z_position_after <= 0.8:
            high_reward = 0.0
        else:
            high_reward = self._height_reward_weight * (z_position_after - 0.8)**2 #np.exp(-0.5*height_error**2/0.2**2)
        low_reward = self._height_reward_weight * (1.3-z_position_after)**2 * np.sign(1.3-z_position_after)

        rewards = 0.0
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            'reward_0': fast_forward_reward + healthy_reward,
            'reward_1': slow_forward_reward + healthy_reward,
            'reward_2': clipped_forward_reward + high_reward,
            'reward_3': clipped_forward_reward + 2*low_reward + healthy_reward*0.5
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
