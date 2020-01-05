import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.utils import q_inv, q_mult 
import xml.etree.ElementTree as ET
from os import path


class AntCrossMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_v2_cross_maze.xml',
                 ctrl_cost_weight=0.0,
                 contact_cost_weight=0.0,
                 healthy_reward=0.0,
                 dead_cost_weight=-100.0,
                 terminate_when_unhealthy=True,
                 alive_z_range=(0.2,1.0),
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 terminate_when_goal_reached=True,
                 goal_radius=0.75,
                 goal_reward_weight=1.0,
                 velocity_reward_weight=0.0,
                 erratic_cost_weight=0.0,
                 velocity_deviation_cost_weight=0.0,
                 possible_goal_positions=[[8.4, 8.4, 0], [8.4, -8.4, 0], [16.8, 0, 0]],
                 n_rays=20,
                 sensor_span=np.pi*0.8,
                 sensor_range=5,
                 verbose=0, 
                 rgb_rendering_tracking=True,
                 save_init_quaternion=False):
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

        self._velocity_reward_weight = velocity_reward_weight
        self._erratic_cost_weight = erratic_cost_weight
        self._velocity_deviation_cost_weight = velocity_deviation_cost_weight

        self._possible_goal_positions = possible_goal_positions
        self._goal_ind = random.randint(0, len(self._possible_goal_positions)-1)
        self._goal_position = np.array(self._possible_goal_positions[self._goal_ind])[:2]
        
        '''
        A different indicator will be given to each kind of geom in the model: 
        0 for  the ant body, 1 for the walls and floor, and 2 for the target.
        The order of the geoms can be obtained with self.model.geom_names:
        ('floor', 'wall-0', 'wall-1', 'wall-2', 'wall-3', 'wall-4', 'wall-5', 'wall-6', 'wall-7', 'wall-8', 'wall-9', 'wall-10', 'wall-11', 'target',
         'torso_geom', 'aux_1_geom', 'left_leg_geom', 'left_ankle_geom', 'aux_2_geom', 'right_leg_geom', 'right_ankle_geom',   
         'aux_3_geom', 'back_leg_geom', 'third_ankle_geom', 'aux_4_geom', 'rightback_leg_geom', 'fourth_ankle_geom')
        '''        
        self._obstacle_types = [0,1,1,1,1,1,1,1,1,1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0]
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

        self._init_rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._save_init_quaternion = save_init_quaternion
                
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._target_in_sight = False
        self._verbose = verbose

        self._xml_file = xml_file
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self._update_goal_visualization()

    def _update_goal_visualization(self):
        goal_id = self.model.geom_names.index('target')
        self.model.geom_pos[goal_id] = np.asarray(self._possible_goal_positions[self._goal_ind], dtype=np.float64)

    @property
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward

    @property
    def dead_cost(self):
        return float(not self.is_alive) * self._dead_cost_weight

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def xy_velocity(self):
        return self.sim.data.qvel.flat.copy()[:2]

    @property    
    def velocity_reward(self):        
        velocity_reward = self._velocity_reward_weight * np.dot(self.xy_velocity, self.xy_orientation)
        return velocity_reward

    @property
    def velocity_deviation_cost(self):
        velocity_deviation_cost = 0.5 * self._velocity_deviation_cost_weight * (self.sim.data.qvel.flat[2:].copy()**2).sum()
        return velocity_deviation_cost

    def erratic_cost(self, past_xy_velocity):
        xy_velocity = self.sim.data.qvel.flat.copy()[:2]
        coherence = np.dot(past_xy_velocity, xy_velocity)
        coherence = np.sign(coherence)*np.absolute(coherence)**0.5
        erratic_cost = -coherence * self._erratic_cost_weight
        # erratic_cost = (-np.min([0, coherence]))**0.5 * self._erratic_cost_weight
        return erratic_cost

    @property
    def goal_reached(self):
        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self._goal_position)
        goal_reached = goal_distance < self._goal_radius        
        return goal_reached

    @property
    def goal_reward(self):
        goal_reward = int(self.goal_reached) * self._goal_reward_weight
        # goal_reward = 0.5*(goal_distance/self._goal_radius + 1) * self._goal_reward_weight * float(self.goal_reached)
        return goal_reward

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext.flat.copy()
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))
        return contact_cost

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
    def center_of_mass(self):
        idx = self.model.body_names.index("torso")
        return self.sim.data.subtree_com[idx]


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
                elif self._obstacle_types[obstacle_id] == 2:
                    self._goal_readings[ray] = (self._sensor_range - distance) / self._sensor_range
                    self._goal_sizes[ray] = 1.0
                elif self._obstacle_types[obstacle_id] == 3:
                    danger_readings[ray] = (self._sensor_range - distance) / self._sensor_range            

        obs = np.concatenate([
            wall_readings.copy(),
            self._goal_readings.copy(),
            self._goal_sizes.copy()# ,
            # danger_readings.copy()
        ])
        
        self._target_in_sight = self._goal_readings.sum() > 0.0

        
        return obs

    # @property
    # def sensor_reward(self):

    def step(self, action):
        # past_xy_velocity = self.sim.data.qvel.flat[:2].copy()
        self.do_simulation(action, self.frame_skip)        
        
        velocity_reward = self.velocity_reward
        goal_reward = self.goal_reward
        healthy_reward = self.healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        dead_cost = self.dead_cost
        # erratic_cost = self.erratic_cost(past_xy_velocity)
        # velocity_deviation_cost = self.velocity_deviation_cost

        rewards = velocity_reward + healthy_reward + goal_reward        
        costs = ctrl_cost + contact_cost + dead_cost
        reward = rewards - costs - 1.0

        if self._verbose > 0:
            maze_obs = self.get_current_maze_obs()
            print("Current wall obs: "+str(maze_obs[:self._n_rays]))
            print("Current goal obs: "+str(maze_obs[self._n_rays:]))

        if self._verbose > 1:
            print('healthy_reward: '+str(np.around(healthy_reward,decimals=4))+', '+'velocity_reward: '+str(np.around(velocity_reward,decimals=2))+', '+'goal_reward: '+str(np.around(goal_reward,decimals=2)))
            print( 'ctrl_cost: '+str(np.around(ctrl_cost,decimals=2))+', '+'contact_cost: '+str(np.around(contact_cost,decimals=2))+', '+'erratic_cost: '+str(np.around(erratic_cost,decimals=2))+', '+
                    'dead_cost: '+str(np.around(dead_cost,decimals=2))+', '+'velocity_deviation_cost: '+str(np.around(velocity_deviation_cost,decimals=2)) )
            
        done = self.done
        info = dict(
            reward_goal=goal_reward,
            goal_ind=self._goal_ind)
        observation = self._get_obs()

        # Return
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

        self._goal_ind = random.randint(0, len(self._possible_goal_positions)-1)
        self._goal_position = np.array(self._possible_goal_positions[self._goal_ind])[:2]

        self._update_goal_visualization()

        self._goal_readings = np.zeros(self._n_rays)
        self._goal_sizes = np.zeros(self._n_rays)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
