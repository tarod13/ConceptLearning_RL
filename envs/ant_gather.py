import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.utils import q_inv, q_mult


DEFAULT_CAMERA_CONFIG = {
    'distance': 16.0,
}


class AntGatherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_v2_square_green.xml',
                 ctrl_cost_weight=5e-3,
                 contact_cost_weight=1e-4,
                 healthy_reward=0.0,
                 dead_cost_weight=100,
                 terminate_when_unhealthy=True,	
		 		 alive_z_range=(0.2,1.0),
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 velocity_reward_weight=1.0,
                 exclude_current_positions_from_observation=False,
                 n_rays=20,
                 sensor_span=0.8*np.pi,
                 sensor_range=6,
                 save_init_quaternion=True,
                 catch_reward_weight=20.0,
                 agent_object_spacing=3,
                 object_object_spacing=3.5,
                 room_length=21,
                 object_radius=0.95,
                 catch_range=0.75,
                 n_targets=12,
                 n_bombs=0,
                 n_steps_target_depletion=40):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._dead_cost_weight = dead_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._alive_z_range = alive_z_range

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

        self._obstacle_types = [0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             

        self._catch_reward_weight = catch_reward_weight
        self._min_orientation_similarity = np.cos(0.6*np.pi*0.5)
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

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5) 
        self._obtain_ids()
        self._reset_objects()
    
    def _generate_plaussible_positions(self):
        p = np.zeros((self._n_objects,3))
        p[:,2] = np.ones(self._n_objects)*0.85
        for i in range(0,self._n_objects):
            while not ((((p[i,:2] - p[:i,:2])**2).sum(1) > self._object_object_spacing**2).all() and (p[i,:2]**2).sum() > self._agent_object_spacing**2):
                p[i,:2] = (np.random.rand(2)-0.5)*(self._room_length-2.5-3*self._object_radius)
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
            if self._objects_ON[i] >= 1.0 and self.in_catch_range[i] and self._target_in_sight:
                self._objects_ON[i] -= 1.0
            elif self._objects_ON[i] < 1.0:
                self.model.geom_pos[self._object_ids[i]] = -50*np.asarray([1.0,1.0,1.0], dtype=np.float64)
            self.model.geom_rgba[self._object_ids[i], 3] = self._objects_ON[i]/self._n_steps_target_depletion

    @property
    def in_outter_circle(self):
        squared_distances = ((self._object_positions - self.head_position.reshape(1,-1))[:,:2]**2).sum(1)
        in_outter_circle = squared_distances <= (self._object_radius-0.1)**2
        return in_outter_circle

    @property
    def in_catch_range(self):
        # squared_distance = ((self._goal_position[:2] - self.head_position[:2])**2).sum()
        # in_inner_circle =  squared_distance <= self._catch_range**2
        # in_outter_circle = squared_distance <= (self._goal_radius-0.1)**2
        return self.in_outter_circle & self.in_angle_range

    @property
    def orientation_similarity(self):
        v = (self._object_positions - self.body_position.reshape(1,-1))[:,:2]
        v /= ((v**2).sum(1, keepdims=True))**0.5
        # object_angles = np.arctan2(v[:,1],v[:,0])
        # object_directions = np.zeros([object_angles.shape[0], 2])
        # for i, theta in enumerate(object_angles):
        #     object_directions[i,:] = np.array(self.ray_orientation(theta)[:2])
        # object_directions /= ((object_directions**2).sum(1, keepdims=True))**0.5
        orientation_similarity = (v * self.xy_orientation.reshape(1,-1)).sum(1)
        return orientation_similarity

    @property
    def in_angle_range(self):
       return self.orientation_similarity >= self._min_orientation_similarity

    @property
    def gathering_reward(self):
        return np.dot(self.in_catch_range.copy(), self._objects_ON.copy().clip(0.0,1.0)) * self._catch_reward_weight * self._target_in_sight

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def is_alive(self):
        state = self.state_vector()
        min_z, max_z = self._alive_z_range
        is_alive = (np.isfinite(state).all() and min_z <= state[2] <= max_z and
                        np.sqrt(2.0)*np.abs(self.orientation[2])<1.0 and self.orientation_z[2] >= 0.5)
        return is_alive

    @property
    def finished_gathering(self):
        # return (self._objects_ON[0:self._n_targets]<1.0).all()
        return False

    @property
    def done(self):
        done = (not self.is_alive and self._terminate_when_unhealthy) or self.finished_gathering                
        return done

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
                    self._goal_readings[ray] = (self._sensor_range - distance) / self._sensor_range + 0.5*int(self.in_outter_circle[obstacle_id-5])*(self.orientation_similarity[obstacle_id-5] + 1.0) # * self.in_catch_range[obstacle_id-5]
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
        return float(not self.is_alive) *  self._dead_cost_weight

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = raw_contact_forces / 40 # np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    def step(self, action):
        xy_velocity_before = self.xy_velocity.copy()
        self.do_simulation(action, self.frame_skip)
        xy_velocity_after = self.xy_velocity.copy()
        # observation = self._get_obs()
        wall_observation = self.get_current_maze_obs()[:self._n_rays] # observation[:self._n_rays]
        wall_near = wall_observation.max() > 0.95
        collision_detected = wall_near and np.dot(xy_velocity_before, xy_velocity_after) < 0.2 and (xy_velocity_before**2).sum() > 0.1   
        xy_acceleration = ((xy_velocity_after - xy_velocity_before)**2).sum() / self.dt
        
        ctrl_cost = self.control_cost(action)        
        dead_cost = self.dead_cost
        collision_cost = min((int(collision_detected) * xy_acceleration), 100)
        # contact_cost = self.contact_cost + (xy_acceleration**2).sum()

        forward_reward = self.velocity_reward()
        healthy_reward = self.healthy_reward
        gathering_reward = self.gathering_reward
        
        rewards = healthy_reward + forward_reward + gathering_reward
        costs = ctrl_cost + dead_cost + collision_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {}

        self._update_objects()

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        # maze_obs = self.get_current_maze_obs()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, self._init_quaternion)) # , maze_obs

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

        self._reset_objects()
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
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] += 0
        self.viewer.cam.lookat[1] -= 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.elevation = -55


    # def position_in_initial_frame(self):
    #     position = self.sim.data.qpos.flat.copy()
    #     quaternion = position[3:7]
    #     quaternion_in_local_frame = q_mult(q_inv(self._init_quaternion), quaternion)
    #     position[3:7] = quaternion_in_local_frame
    #     return position
    
    # def velocity_in_initial_frame(self):
    #     velocity = self.sim.data.qvel.flat.copy()
    #     xy_velocity = velocity[0:2]
    #     speed = np.dot(xy_velocity, xy_velocity)**0.5
    #     velocity_direction = xy_velocity / speed
    #     xy_velocity_quaternion = np.array([0.0, velocity_direction[0], velocity_direction[1], 0.0])
    #     quaternion_in_local_frame = q_mult(q_mult(q_inv(self._init_quaternion), xy_velocity_quaternion) , self._init_quaternion)
    #     velocity[0:2] = speed * np.array(quaternion_in_local_frame[1:3])
    #     return velocity
    
    # def _update_quaternion(self):
    #     self._init_quaternion = self.sim.data.qpos.flat.copy()[3:7]

    # def _get_obs(self):
    #     # position = self.sim.data.qpos.flat.copy()
    #     position = self.position_in_initial_frame()
    #     velocity = self.velocity_in_initial_frame()
    #     maze_obs = self.get_current_maze_obs()
    #     # contact_force = self.contact_forces.flat.copy()  

    #     # print("Maze obs:")
    #     # print(np.around(maze_obs*10,1))
    #     # print("Contact force:")
    #     # print(contact_force)

    #     # angle = self._init_angle if self._save_init_angle else self.xy_orientation_angle
    #     # quaternion = self._init_quaternion if self._save_init_quaternion else position[3:7]
    #     # target = self._target if self._save_init_angle else self._D_max * np.array([np.cos(angle), np.sin(angle)])
    #     # target_in_head_frame = target - self.head_position[:2]

    #     if self._exclude_current_positions_from_observation:
    #         position = position[2:]

    #     observations = np.concatenate((position, velocity, maze_obs)) # np.array([angle]), 

    #     return observations
