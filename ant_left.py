import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class AntLeftEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_random_direction.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.3, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 velocity_reward_weight=1.0e-0,
                 velocity_deviation_cost_weight=0.0,
                 proper_orientation_weight=0.0,
                 exclude_current_positions_from_observation=True,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._velocity_reward_weight = velocity_reward_weight
        self._velocity_deviation_cost_weight = velocity_deviation_cost_weight
        self._proper_orientation_weight = proper_orientation_weight

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._possible_angles = np.linspace(0., 2*np.pi, 3)[:-1] + np.pi/2
        self._task = 0
        self._angle = self._possible_angles[self._task]
        self._rgba_ON = np.asarray([0., 1., 0., 0.3], dtype=np.float32)
        self._rgba_OFF = np.asarray([0., 0.2, 0., 0.2], dtype=np.float32)
        
        self._goal_direction = np.array([np.cos(self._angle), np.sin(self._angle)])

        self._eps = 1e-6

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

        self._update_goal_visualization()

    def _update_goal_visualization(self):
        indicator_id1 = self.model.geom_names.index('indicator1')
        indicator_id2 = self.model.geom_names.index('indicator2')
        self.model.geom_rgba[indicator_id1] = self._rgba_ON
        self.model.geom_rgba[indicator_id2] = self._rgba_OFF
        
    @property
    def orientation(self):
        a, vx, vy, vz = self.sim.data.qpos[3:7].copy() # rotation quaternion (roll-pitch-yaw)
        orientation = [1-2*(vy**2+vz**2), 2*(vx*vy + a*vz), 2*(vx*vz - a*vy)]        
        return orientation

    @property
    def xy_orientation(self):
        orientation = np.array(self.orientation[:2])
        orientation /= np.dot(orientation,orientation)
        return orientation

    @property    
    def velocity_reward(self):
        xy_velocity = self.sim.data.qvel.flat.copy()[:2]
        velocity_reward = self._velocity_reward_weight * np.dot(xy_velocity, self.xy_orientation)
        return velocity_reward

    @property
    def velocity_deviation_cost(self):
        velocity_deviation_cost = 0.5 * self._velocity_deviation_cost_weight * (self.sim.data.qvel.flat[2:].copy()**2).sum()
        return velocity_deviation_cost

    @property
    def orientation_reward(self):
        reward = self._proper_orientation_weight * np.dot(self.xy_orientation, self._goal_direction)
        return reward

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
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        velocity_deviation_cost = self.velocity_deviation_cost

        forward_reward = np.dot(xy_velocity, self._goal_direction) #- np.absolute(np.dot(xy_velocity, self._goal_direction[[1,0]]))*0.5
        orientation_reward = self.orientation_reward
        # forward_reward = self.velocity_reward 
        healthy_reward = self.healthy_reward
        # goal_reward =  forward_reward #np.dot(xy_velocity, self._goal_direction), orientation_reward

        rewards = orientation_reward + forward_reward + healthy_reward 
        costs = ctrl_cost + contact_cost + velocity_deviation_cost

        reward = rewards - costs
        goal_reward = reward        
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,
            'reward_goal': goal_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
            'goal_direction': self._angle,    
            'task': self._task,         
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
