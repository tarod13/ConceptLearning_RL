import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.utils import q_inv, q_mult


DEFAULT_CAMERA_CONFIG = {
    'distance': 16.0,
}


class AntGatherBEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_v2_square_green.xml',
                 ctrl_cost_weight=0,
                 contact_cost_weight=0.0,
                 healthy_reward=0.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-10.0, 10.0),
                 reset_noise_scale=0.1,
                 velocity_reward_weight=1.0,
                 catch_reward_weight=-20.0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True,
                 n_rays=20,
                 sensor_span=0.8*np.pi,
                 sensor_range=5,
                 agent_object_spacing=2,
                 object_object_spacing=3.5,
                 room_length=21,
                 object_radius=0.75,
                 catch_range=0.95,
                 n_targets=16,
                 n_bombs=0,
                 n_steps_target_depletion=40,
                 save_init_quaternion=False):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._velocity_reward_weight = velocity_reward_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._catch_reward_weight = catch_reward_weight

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._obstacle_filter = np.asarray([False, True], dtype=np.uint8)        
        self._n_rays = n_rays
        self._sensor_span = sensor_span
        self._min_orientation_similarity = np.cos(sensor_span*0.5)
        self._sensor_range = sensor_range
        self._ray_angles = np.zeros(n_rays)
        for ray in range(self._n_rays):
            self._ray_angles[ray] = self._sensor_span * (- 0.5 + (2*ray + 1)/(2*self._n_rays))
        self._goal_readings = np.zeros(n_rays)
        self._goal_sizes = np.zeros(n_rays)

        self._init_rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._save_init_quaternion = save_init_quaternion

        self._obstacle_types = [0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             

        self._agent_object_spacing = agent_object_spacing
        self._object_object_spacing = object_object_spacing
        self._room_length = room_length
        self._object_radius = object_radius
        self._catch_range = catch_range 
        self._n_targets = n_targets
        self._n_bombs = n_bombs
        self._n_objects = n_targets + n_bombs
        self._n_steps_target_depletion = n_steps_target_depletion*1.0
        self._objects_ON = np.ones(self._n_objects)*n_steps_target_depletion
        # self._reward_mask = np.array(n_targets*[1] + n_bombs*[-1])
        self._object_positions = 10.0*np.ones((self._n_objects,3)) 
        self._object_ids = np.array(self._n_objects*[0])
        self._target_in_sight = False

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking) 
        self._obtain_ids()
        self._reset_objects()
    
    def _generate_plaussible_positions(self):
        p = np.zeros((self._n_objects,3))
        p[:,2] = np.ones(self._n_objects)*0.85
        for i in range(0,self._n_objects):
            while not ((((p[i,:2] - p[:i,:2])**2).sum(1) > self._object_object_spacing**2).all() and (p[i,:2]**2).sum() > self._agent_object_spacing**2):
                p[i,:2] = (np.random.rand(2)-0.5)*(self._room_length-2.0-3*self._object_radius)
        self._object_positions = p.copy()

    def _obtain_ids(self):
        for i in range(0,self._n_targets):
            self._object_ids[i] = self.model.geom_names.index('target_'+str(i))
        for i in range(0,self._n_bombs):
            self._object_ids[i+self._n_targets] = self.model.geom_names.index('bomb_'+str(i))

    def _reset_objects(self):
        self._objects_ON = np.ones(self._n_objects)*self._n_steps_target_depletion   
        self._generate_plaussible_positions()
        for i in range(0,self._n_targets):
            self.model.geom_pos[self._object_ids[i]] = np.asarray(self._object_positions[i,:], dtype=np.float64)
            self.model.geom_rgba[self._object_ids[i], 3] = 1.0
        # for i in range(0,self._n_bombs):
        #     self._objects_ON[i+self._n_targets] = 0.0
        #     self.model.geom_pos[self._object_ids[i+self._n_targets]] = np.asarray(self._object_positions[i+self._n_targets,:], dtype=np.float64)
        #     self.model.geom_rgba[self._object_ids[i+self._n_targets], 3] = 0.0
    
    def _update_objects(self):
        for i in range(0, self._n_objects):
            if self._objects_ON[i] >= 1.0 and self.in_catch_range[i]:
                self._objects_ON[i] -= 1.0
            elif self._objects_ON[i] < 1.0:
                self.model.geom_pos[self._object_ids[i]] = -50*np.asarray([1.0,1.0,1.0], dtype=np.float64)
            self.model.geom_rgba[self._object_ids[i], 3] = self._objects_ON[i]/self._n_steps_target_depletion

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
        orientation_quaternion = [0.0, np.cos(theta), np.sin(theta), 0.0]
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
            self._goal_sizes.copy()# ,
            # danger_readings.copy()
        ])
        
        self._target_in_sight = self._goal_readings.sum() > 0.0

        return obs

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

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
    def in_catch_range(self):
        return ((self._object_positions - self.body_position.reshape(1,-1))**2).sum(1) <= self._catch_range**2

    @property
    def in_angle_range(self):
        v = self._object_positions - self.body_position.reshape(1,-1)
        v /= ((v**2).sum(1, keepdims=True))**0.5
        orientation_similarity = (v[:,:2] * self.xy_orientation.reshape(1,-1)).sum(1)        
        return orientation_similarity >= self._min_orientation_similarity

    @property
    def gathering_reward(self):
        return np.dot(self.in_catch_range.copy()*self.in_angle_range.copy()*self._target_in_sight, self._objects_ON.copy().clip(0.0,1.0)) * self._catch_reward_weight

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

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        min_xy, max_xy = -self._room_length/2, self._room_length/2
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z and 
                        min_xy <= state[0] <= max_xy and min_xy <= state[1] <= max_xy and
                        np.sqrt(2.0)*np.abs(self.orientation[2])<1.0 and self.orientation_z[2] >= 0.5)
        return is_healthy

    @property
    def finished_gathering(self):
        # return (self._objects_ON[0:self._n_targets]<1.0).all()
        return False

    @property
    def done(self):
        done = (not self.is_healthy and self._terminate_when_unhealthy) or self.finished_gathering                
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        velocity_reward = self.velocity_reward
        gathering_reward = self.gathering_reward
        healthy_reward = self.healthy_reward        

        rewards = gathering_reward + healthy_reward + velocity_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        goal_reward =  reward
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_gathering': gathering_reward,
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

        self._update_objects()

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

        self._reset_objects()
        self._goal_readings = np.zeros(self._n_rays)
        self._goal_sizes = np.zeros(self._n_rays)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.elevation = -55
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[2] = 0
