import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.utils import q_inv, q_mult


DEFAULT_CAMERA_CONFIG = {
    'distance': 30.0,
    'elevation': -55,
    'lookat': np.array([6.5,0,0])
}


class AntSquareTrackEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_v2_square_track.xml',
                 ctrl_cost_weight=0,
                 contact_cost_weight=0.0,
                 healthy_reward=0.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-10.0, 10.0),
                 reset_noise_scale=0.1,
                 velocity_reward_weight=1.0e-0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True,
                 n_rays=20,
                 sensor_span=np.pi*0.8,
                 sensor_range=5,
                 save_init_quaternion=False):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._velocity_reward_weight = velocity_reward_weight

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._obstacle_filter = np.asarray([False, True], dtype=np.uint8)        
        self._n_rays = n_rays
        self._sensor_span = sensor_span
        self._sensor_range = sensor_range
        self._ray_angles = np.zeros(n_rays)
        for ray in range(self._n_rays):
            self._ray_angles[ray] = self._sensor_span * (- 0.5 + (2*ray + 1)/(2*self._n_rays))
        self._goal_readings = np.zeros(n_rays)
        self._goal_sizes = np.zeros(n_rays)

        self._init_rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._save_init_quaternion = save_init_quaternion 

        self._obstacle_types = [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)  
        
    @property
    def orientation(self):
        a, vx, vy, vz = self.sim.data.qpos[3:7].copy() # rotation quaternion (roll-pitch-yaw)
        orientation = [1-2*(vy**2+vz**2), 2*(vx*vy + a*vz), 2*(vx*vz - a*vy)]
        orientation /= np.dot(orientation,orientation)**0.5        
        return orientation

    @property
    def orientation_z(self):
        a, vx, vy, vz = self.sim.data.qpos[3:7].copy() # rotation quaternion (roll-pitch-yaw)
        orientation = [2*(vx*vz - a*vy), 2*(vy*vz + a*vx), 1-2*(vx**2+vy**2)]
        orientation /= np.dot(orientation,orientation)**0.5        
        return orientation

    @property
    def xy_orientation(self):
        orientation = np.array(self.orientation[:2])
        orientation /= np.dot(orientation,orientation)**0.5
        return orientation

    def ray_orientation(self, theta):
        orientation_quaternion = [0, np.cos(theta), np.sin(theta), 0]
        rotation_quaternion = self.sim.data.qpos[3:7].copy()
        orientation = q_mult(q_mult(rotation_quaternion, orientation_quaternion), q_inv(rotation_quaternion))[1:]
        return orientation

    @property
    def head_position(self):
        # local_head_position = np.asarray(self.model.geom_pos[self.model.geom_names.index('head_geom')], dtype=np.float64)
        # global_head_position = self.body_position + local_head_position
        # return global_head_position
        return np.asarray(self.get_body_com("head")[:3], dtype=np.float64)
    
    @property
    def body_position(self):
        return np.asarray(self.get_body_com("torso")[:3], dtype=np.float64)

    def get_current_maze_obs(self):   
        wall_readings = np.zeros(self._n_rays)
        self._goal_readings = np.zeros(self._n_rays)
        self._goal_sizes = np.zeros(self._n_rays)
        danger_readings = np.zeros(self._n_rays)

        for ray in range(self._n_rays):
            ray_theta = self._ray_angles[ray]
            ray_orientation = np.asarray(self.ray_orientation(ray_theta), dtype=np.float64)
            distance, obstacle_id = self.sim.ray_fast_group(self.head_position, ray_orientation, self._obstacle_filter)           
            if obstacle_id >= 0 and distance <= self._sensor_range:
                if self._obstacle_types[obstacle_id] == 1:
                    wall_readings[ray] = (self._sensor_range - distance) / self._sensor_range
                elif self._obstacle_types[obstacle_id] == 2 and self._objects_ON[obstacle_id-5] >= 1.0:
                    self._goal_readings[ray] = (self._sensor_range - distance) / self._sensor_range
                    self._goal_sizes[ray] = self._objects_ON[obstacle_id-5] / self._n_steps_target_depletion
                elif self._obstacle_types[obstacle_id] == 3 and self._objects_ON[obstacle_id-5] >= 1.0:
                    danger_readings[ray] = (self._sensor_range - distance) / self._sensor_range            

        obs = np.concatenate([
            wall_readings.copy(),
            self._goal_readings.copy(),
            self._goal_sizes.copy()#,
            #danger_readings.copy()
        ])
        return obs

    @property
    def xy_position(self):
        return self.get_body_com("torso")[:2].copy()

    @property
    def xy_velocity(self):
        return self.sim.data.qvel.flat.copy()[:2]

    @property    
    def velocity_reward(self):
        speed = np.dot(self.xy_velocity, self.xy_velocity)**0.5
        velocity_direction = self.xy_velocity / speed
        similarity = np.dot(velocity_direction, self.xy_orientation)
        velocity_reward = self._velocity_reward_weight * speed * np.sign(similarity)*similarity**2
        return velocity_reward

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
            np.square(self.contact_forces[:,:5]))
        return contact_cost

    # @property
    # def is_inside(self):
    #     return not self.zone == 0

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z  and np.sqrt(2.0)*np.abs(self.orientation[2])<1.0 and self.orientation_z[2] >= 0.5) # and self.is_inside
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xy_position_before = self.xy_position
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.xy_position

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self.velocity_reward
        healthy_reward = self.healthy_reward
        
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

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
            'y_velocity': y_velocity     
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        maze_obs = self.get_current_maze_obs()

        # print("Maze obs:")
        # print(np.around(maze_obs*10,1))
        # print("Contact force:")
        # print(contact_force)

        quaternion = self._init_rotation_quaternion if self._save_init_quaternion else position[3:7]

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, quaternion, maze_obs))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)        
        
        angle = (np.random.rand()-0.5)*2.0*np.pi
        qpos[3] = np.cos(angle/2.0)
        qpos[4] = qpos[5] = 0.0
        qpos[6] = np.sin(angle/2.0)

        self._init_rotation_quaternion = qpos[3:7].copy()

        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        self._goal_readings = np.zeros(self._n_rays)
        self._goal_sizes = np.zeros(self._n_rays)

        observation = self._get_obs()

        return observation

    # def viewer_setup(self):
    #     for key, value in DEFAULT_CAMERA_CONFIG.items():
    #         if isinstance(value, np.ndarray):
    #             getattr(self.viewer.cam, key)[:] = value
    #         else:
    #             setattr(self.viewer.cam, key, value)
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.elevation = -55
        self.viewer.cam.lookat[0] = 6.5
        self.viewer.cam.lookat[2] = 0
