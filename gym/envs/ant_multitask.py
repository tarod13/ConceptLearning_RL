import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.utils import q_inv, q_mult


DEFAULT_CAMERA_CONFIG = {
    'distance': 25.0,
    'trackbodyid': 2
}


class AntMTEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_v2_white.xml',
                 ctrl_cost_weight=5e-3,
                 contact_cost_weight=0,
                 healthy_reward=0.0,
                 dead_cost_weight=100,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.6,
                 velocity_reward_weight=1.0e-0,
                 exclude_current_positions_from_observation=False,
                 n_rays=20,
                 sensor_span=np.pi*0.8,
                 sensor_range=5,
                 save_init_quaternion=True                 
                 ):
        utils.EzPickle.__init__(**locals())

        self._n_tasks = 3
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._dead_cost_weight = dead_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale

        self._velocity_reward_weight = velocity_reward_weight
        
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        self._obstacle_filter = np.asarray([False, True], dtype=np.uint8)        
        self._n_rays = n_rays
        self._sensor_span = sensor_span
        self._sensor_range = sensor_range
        self._ray_angles = np.zeros(n_rays)
        for ray in range(self._n_rays):
            self._ray_angles[ray] = self._sensor_span * (- 0.5 + (2*ray + 1)/(2*self._n_rays))
        self._goal_readings = np.zeros(n_rays)
        self._goal_sizes = np.zeros(n_rays)

        self._init_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._save_init_quaternion = save_init_quaternion
        
        self._obstacle_types = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def orientation(self):
        a, vx, vy, vz = self.sim.data.qpos[3:7].copy() # quaternion frame (roll-pitch-yaw)
        orientation = [1-2*(vy**2+vz**2), 2*(vx*vy + a*vz), 2*(vx*vz - a*vy)]
        orientation /= np.dot(orientation,orientation)**0.5        
        return orientation

    @property
    def orientation_z(self):
        a, vx, vy, vz = self.sim.data.qpos[3:7].copy() # quaternion frame (roll-pitch-yaw)
        orientation = [2*(vx*vz - a*vy), 2*(vy*vz + a*vx), 1-2*(vx**2+vy**2)]
        orientation /= np.dot(orientation,orientation)**0.5        
        return orientation

    @property
    def xy_orientation(self):
        orientation = np.array(self.orientation[:2])
        orientation /= np.dot(orientation,orientation)**0.5
        return orientation.copy()
    
    def xy_orientation_init(self):
        a, vx, vy, vz = self._init_quaternion # quaternion frame (roll-pitch-yaw)
        orientation = [1-2*(vy**2+vz**2), 2*(vx*vy + a*vz), 2*(vx*vz - a*vy)]
        orientation /= np.dot(orientation,orientation)**0.5
        xy_orientation = np.array(orientation[:2])
        xy_orientation /= np.dot(xy_orientation,xy_orientation)**0.5
        return xy_orientation   

    @property
    def xy_orientation_angle(self):
        x, y = self.xy_orientation
        return np.arctan2(y, x)

    def ray_orientation(self, theta):
        orientation_quaternion = [0, np.cos(theta), np.sin(theta), 0]
        rotation_quaternion = self.sim.data.qpos[3:7].copy()
        orientation = q_mult(q_mult(rotation_quaternion, orientation_quaternion), q_inv(rotation_quaternion))[1:]
        return orientation

    @property
    def head_position(self):
        return np.asarray(self.get_body_com("head")[:3], dtype=np.float64).copy()
    
    @property
    def body_position(self):
        return np.asarray(self.get_body_com("torso")[:3], dtype=np.float64).copy()

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
            self._goal_sizes.copy()# ,
            # danger_readings.copy()
        ])
        
        self._target_in_sight = self._goal_readings.sum() > self._n_rays/4

        return obs

    def rotate_vector(self, angle, vector):
        c = np.cos(angle)
        s = np.sin(angle)
        rotation_matrix = np.array([[ c, s],
                      [-s, c]])
        return np.dot(rotation_matrix, vector) 

    @property
    def xy_velocity(self):
        return self.sim.data.qvel.flat.copy()[:2]

    def velocity_reward(self):
        speed = np.dot(self.xy_velocity, self.xy_velocity)**0.5
        velocity_direction = self.xy_velocity / speed
        similarity = np.dot(velocity_direction, self.xy_orientation)
        similarity_2 = np.dot(self.xy_orientation_init(), self.xy_orientation)
        similarity_3 = np.dot(velocity_direction, self.xy_orientation_init())
        reward = np.sign(similarity) * similarity**2
        reward += np.sign(similarity_2) * similarity_2**2
        reward += np.sign(similarity_3) * similarity_3**2
        return self._velocity_reward_weight * speed * reward/3.0

    def angular_velocity_reward(self, orientation_before):
        x_before, y_before = orientation_before
        angle_before = np.arctan2(y_before, x_before)
        x_rotated, y_rotated = self.rotate_vector(angle_before, self.xy_orientation)        
        angle_in_previous_frame = np.arctan2(y_rotated, x_rotated)
        velocity = angle_in_previous_frame/(np.pi * self.dt)
        return self._velocity_reward_weight * velocity
    
    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def dead_cost(self):
        return float(not self.is_healthy) *  self._dead_cost_weight

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
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z and np.sqrt(2.0)*np.abs(self.orientation[2])<1.0 and self.orientation_z[2] >= 0.5)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy and self._terminate_when_unhealthy) # or self.target_reached
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        xy_orientation_before = self.xy_orientation
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)        
        dead_cost = self.dead_cost
        # contact_cost = self.contact_cost

        forward_reward = self.velocity_reward()
        anticlock_reward = self.angular_velocity_reward(xy_orientation_before)
        linear_velocity_cost = (xy_velocity**2).sum() * self._velocity_reward_weight
        healthy_reward = self.healthy_reward
        
        rewards = healthy_reward
        costs = ctrl_cost + dead_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_0': forward_reward-np.abs(anticlock_reward),
            'reward_2': anticlock_reward-linear_velocity_cost,
            'reward_1': -anticlock_reward-linear_velocity_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        maze_obs = self.get_current_maze_obs()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, self._init_quaternion, maze_obs)) 

        return observations
    
    def _update_quaternion(self):
        self._init_quaternion = self.sim.data.qpos[3:7].copy()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)         
        
        angle =  2*np.pi*(np.random.rand()-0.5)
        qpos[3] = np.cos(angle/2.0)
        qpos[4] = qpos[5] = 0.0
        qpos[6] = np.sin(angle/2.0)
        qpos[2] = self.init_qpos[2]

        qpos += self.np_random.uniform(
            low=noise_low/4, high=noise_high/4, size=self.model.nq)

        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        self._init_quaternion = self.sim.data.qpos[3:7].copy()
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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 4
        self.viewer.cam.elevation = -55
