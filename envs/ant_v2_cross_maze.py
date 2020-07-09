import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.utils import q_inv, q_mult 
import xml.etree.ElementTree as ET
from os import path


class AntCrossMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_v2_cross_maze_no_lights.xml',
                 ctrl_cost_weight=5e-3,
                 contact_cost_weight=5e-4,
                 healthy_reward=0.0,
                 dead_cost_weight=100.0,
                 terminate_when_unhealthy=True,
                 alive_z_range=(0.2,1.0),
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 velocity_reward_weight=1.0,
                 exclude_current_positions_from_observation=False,
                 n_rays=20,
                 sensor_span=np.pi*0.8,
                 sensor_range=6,
                 save_init_quaternion=True,
                 goal_reward_weight=100.0,
                 goal_radius=0.95,
                 catch_range=0.75,
                 terminate_when_goal_reached=True,  
                 possible_goal_positions=[[8.4, 8.2, 0.85], [8.4, -8.2, 0.85], [16.6, 0, 0.85], [8.4, 0, 0.85]],
                 easy_start = True,
                 easy_epsds = 0,                 
                 verbose=0
                 ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._dead_cost_weight = dead_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._alive_z_range = alive_z_range
        self._terminate_when_goal_reached = terminate_when_goal_reached

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._goal_radius = goal_radius
        self._goal_reward_weight = goal_reward_weight
        self._catch_range = catch_range
        self._min_orientation_similarity = np.cos(0.6*np.pi*0.5)

        self._velocity_reward_weight = velocity_reward_weight

        self._possible_goal_positions = possible_goal_positions
        self._easy_start = easy_start
        self._easy_epsds = easy_epsds
        if easy_start:
            self._goal_ind = len(self._possible_goal_positions)-1
            self._goal_position = np.array(self._possible_goal_positions[-1])[:2]
        else:
            self._goal_ind = np.random.randint(0, len(self._possible_goal_positions)-1)
            self._goal_position = np.array(self._possible_goal_positions[self._goal_ind])[:2]
        self._easy_counter = 0
        
        '''
        A different indicator will be given to each kind of geom in the model: 
        0 for  the ant body, 1 for the walls and floor, and 2 for the target.
        The order of the geoms can be obtained with self.model.geom_names:
        ('floor', 'wall-0', 'wall-1', 'wall-2', 'wall-3', 'wall-4', 'wall-5', 'wall-6', 'wall-7', 'wall-8', 'wall-9', 'wall-10', 'wall-11', 'target',
         'torso_geom', 'aux_1_geom', 'left_leg_geom', 'left_ankle_geom', 'aux_2_geom', 'right_leg_geom', 'right_ankle_geom',   
         'aux_3_geom', 'back_leg_geom', 'third_ankle_geom', 'aux_4_geom', 'rightback_leg_geom', 'fourth_ankle_geom')
        '''        
        self._obstacle_types = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0]
        '''
        The obstacle filter is an array of Booleans whose size is the same as the geom groups defined in the .xml file.
        In this case, the geom group 0 corresponds to the ant body, the geom group 1 to the floor and the walls, and the geom group 2 to the target  
        '''
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
                
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._target_in_sight = False
        self._verbose = verbose
        self.S_old = 0
        self.A_old = 0

        # self._color_map = [11,3,11,8,7,11,11,2,11,0,5,4,11,11,9,1,11,10,11,6]

        self._led_colors = np.array([
                                        [0.6784313725490196, 0.28627450980392155, 0.2901960784313726, 1.0],     # C10
                                        [0.38823529411764707, 0.4745098039215686, 0.2235294117647059, 1.0],     # C12
                                        [0.7764705882352941, 0.8588235294117647, 0.9372549019607843, 1.0,],
                                        [0.7764705882352941, 0.8588235294117647, 0.9372549019607843, 1.0,],
                                        [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0],       # 'C4'
                                        [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0],     # C0'
                                        [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0],     # 'C9'
                                        [0.10588235294117647, 0.6196078431372549, 0.4666666666666667, 1.0,],    # C14
                                        [1.0, 0.4980392156862745, 0.054901960784313725, 1.0],                   # 'C1'
                                        [0.6470588235294118, 0.3176470588235294, 0.5803921568627451, 1.0,],     # C15
                                        [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0],    # 'C5'
                                        [0.2235294117647059, 0.23137254901960785, 0.4745098039215686, 1.0,],    # C11
                                        [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0,],     # 'C7'
                                        [0.7411764705882353, 0.6196078431372549, 0.2235294117647059, 1.0,],     # C13
                                        [0.7764705882352941, 0.8588235294117647, 0.9372549019607843, 1.0],
                                        [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0,],    # 'C8'
                                        [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0,],    # 'C3'
                                        [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0,],   # 'C2'
                                        [0.0, 0.0, 0.0, 1.0],                                                   # 'k'
                                        [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0]]      # 'C6'
                                )

        self._led_colors_2 = np.array([     [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0],
                                            [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0],
                                            [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0],
                                            [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0,]]
                                    )

        self._xml_file = xml_file
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self._update_goal_visualization()

    def _update_goal_visualization(self):
        goal_id = self.model.geom_names.index('target')
        self.model.geom_pos[goal_id] = np.asarray(self._possible_goal_positions[self._goal_ind], dtype=np.float64)

    # def _reset_leds(self):
    #     for i in range(0,12):
    #         led_id = self.model.geom_names.index('led-'+str(i))
    #         self.model.geom_rgba[led_id] = self._led_colors[i,:]/5.0
    #     for i in range(0,4):
    #         led_id = self.model.geom_names.index('led2-'+str(i))
    #         self.model.geom_rgba[led_id] = self._led_colors_2[i,:]/5.0

    # def _update_led_visualization(self, S, A):
    #     if S != self.S_old:
    #         led_id = self.model.geom_names.index('led-'+str(S))
    #         self.model.geom_rgba[led_id] = self._led_colors[S,:]
    #         led_id_old = self.model.geom_names.index('led-'+str(self.S_old))
    #         self.model.geom_rgba[led_id_old] = self._led_colors[self.S_old,:]/5.0
    #         self.S_old = S
        
    #     if A != self.A_old:
    #         led_id = self.model.geom_names.index('led2-'+str(A))
    #         self.model.geom_rgba[led_id] = self._led_colors_2[A,:]
    #         led_id_old = self.model.geom_names.index('led2-'+str(self.A_old))
    #         self.model.geom_rgba[led_id_old] = self._led_colors[self.A_old,:]/5.0
    #         self.A_old = A

        #     else:
        #         self.model.geom_rgba[led_id] = self._led_colors[i,:]/5.0  
        # for i in range(0,20):
        #     # if S < 12:
        #         # Si = self._color_map[S]
        #     led_id = self.model.geom_names.index('led-'+str(i))
        #     if i == S:
        #         self.model.geom_rgba[led_id] = self._led_colors[i,:]
        #     else:
        #         self.model.geom_rgba[led_id] = self._led_colors[i,:]/5.0   
        # for i in range(0,4):
        #     led_id = self.model.geom_names.index('led2-'+str(i))
        #     if i == A:
        #         self.model.geom_rgba[led_id] = self._led_colors_2[i,:]
        #     else:
        #         self.model.geom_rgba[led_id] = self._led_colors_2[i,:]/5.0        
    
    @property
    def in_outter_circle(self):
        squared_distance = ((self._goal_position[:2] - self.head_position[:2])**2).sum()
        in_outter_circle = squared_distance <= (self._goal_radius-0.1)**2
        return in_outter_circle

    @property
    def goal_reached(self):
        # squared_distance = ((self._goal_position[:2] - self.head_position[:2])**2).sum()
        # in_inner_circle =  squared_distance <= self._catch_range**2
        # in_outter_circle = squared_distance <= (self._goal_radius-0.1)**2
        return self.in_outter_circle & self.in_angle_range

    @property
    def orientation_similarity(self):
        v = self._goal_position[:2] - self.body_position[:2]
        v /= ((v**2).sum())**0.5
        orientation_similarity = np.dot(v, self.xy_orientation)
        # if self.in_outter_circle: 
        #     print(orientation_similarity, self.xy_orientation)
        return orientation_similarity

    @property
    def in_angle_range(self):
       return self.orientation_similarity >= self._min_orientation_similarity

    @property
    def goal_reward(self):
        goal_reward = int(self.goal_reached) * self._goal_reward_weight
        # goal_reward = 0.5*(goal_distance/self._goal_radius + 1) * self._goal_reward_weight * float(self.goal_reached)
        return goal_reward

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
    def done(self):
        done = ((not self.is_alive and self._terminate_when_unhealthy)
                or
                (self.goal_reached and self._terminate_when_goal_reached)) 
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
                elif self._obstacle_types[obstacle_id] == 2:                    
                    self._goal_readings[ray] = (self._sensor_range - distance) / self._sensor_range + 0.5*int(self.in_outter_circle)*(self.orientation_similarity + 1.0) 
                    self._goal_sizes[ray] = 0.9
                elif self._obstacle_types[obstacle_id] == 3:
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

    # @property
    # def contact_forces(self):
    #     raw_contact_forces = self.sim.data.cfrc_ext
    #     min_value, max_value = self._contact_force_range
    #     contact_forces = np.clip(raw_contact_forces, min_value, max_value)
    #     return contact_forces

    # @property
    # def contact_cost(self):
    #     contact_cost = self._contact_cost_weight * np.sum(
    #         np.square(self.contact_forces))
    #     return contact_cost

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        xy_orientation_before = self.xy_orientation
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before)        
        
        velocity_reward = self.velocity_reward()
        angular_velocity_reward = self.angular_velocity_reward(xy_orientation_before)
        goal_reward = self.goal_reward
        healthy_reward = self.healthy_reward

        ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        dead_cost = self.dead_cost
        angular_velocity_cost = np.sqrt((xy_velocity**2).sum()) * self._velocity_reward_weight
        # erratic_cost = self.erratic_cost(past_xy_velocity)
        # velocity_deviation_cost = self.velocity_deviation_cost

        rewards = healthy_reward + goal_reward        
        costs = dead_cost
        reward = rewards - costs
        reward_0 =  velocity_reward - np.abs(angular_velocity_reward)
        reward_1 =  angular_velocity_reward - angular_velocity_cost
        reward_2 =  -angular_velocity_reward - angular_velocity_cost
        

        # if self._verbose > 0:
        #     maze_obs = self.get_current_maze_obs()
        #     print("Current wall obs: "+str(maze_obs[:self._n_rays]))
        #     print("Current goal obs: "+str(maze_obs[self._n_rays:]))

        # if self._verbose > 1:
        #     print('healthy_reward: '+str(np.around(healthy_reward,decimals=4))+', '+'velocity_reward: '+str(np.around(velocity_reward,decimals=2))+', '+'goal_reward: '+str(np.around(goal_reward,decimals=2)))
        #     print( 'ctrl_cost: '+str(np.around(ctrl_cost,decimals=2))+', '+'contact_cost: '+str(np.around(contact_cost,decimals=2))+', '+'erratic_cost: '+str(np.around(erratic_cost,decimals=2))+', '+
        #             'dead_cost: '+str(np.around(dead_cost,decimals=2))+', '+'velocity_deviation_cost: '+str(np.around(velocity_deviation_cost,decimals=2)) )
            
        done = self.done
        info = {
            # 'reward_0': reward_0,
            # 'reward_1': reward_1,
            # 'reward_2': reward_2   
        }
        observation = self._get_obs()

        # v = self._goal_position - self.body_position[:2]
        # v /= ((v**2).sum())**0.5
        # orientation_similarity = np.dot(v, self.xy_orientation)  
        # print("Goal position" + str(self._goal_position))
        # print("Goal position" + str(self._goal_position))
        # print("Goal position" + str(self._goal_position))
        # print(self._min_orientation_similarity)

        # Return
        return observation, reward, done, info

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

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()
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
        
        if self._easy_start and self._easy_counter < self._easy_epsds:
            self._easy_counter += 1            
            self._goal_ind = len(self._possible_goal_positions)-1
            self._goal_position = np.array(self._possible_goal_positions[-1])[:2]
        else:
            self._goal_ind = np.random.randint(0, len(self._possible_goal_positions)-1)
            self._goal_position = np.array(self._possible_goal_positions[self._goal_ind])[:2]

        self._update_goal_visualization()
        # self._reset_leds()

        self._goal_readings = np.zeros(self._n_rays)
        self._goal_sizes = np.zeros(self._n_rays)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        self.viewer.cam.lookat[0] += 6
        self.viewer.cam.lookat[1] -= 2
        self.viewer.cam.distance = self.model.stat.extent * 0.9
        self.viewer.cam.elevation = -45

