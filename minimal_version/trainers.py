import collections
import numpy as np
from buffers import ExperienceFirstLevel, PixelExperienceSecondLevel
from policy_optimizers import First_Level_SAC_PolicyOptimizer, Second_Level_SAC_PolicyOptimizer, Second_Level_SAC_with_baselines_PolicyOptimizer

from utils import cat_state_task, scale_action

import wandb

import cv2
video_folder = '/home/researcher/Diego/Concept_Learning_minimal/videos/'


class First_Level_Trainer:
    def __init__(self, optimizer_kwargs={}):
        self.optimizer = First_Level_SAC_PolicyOptimizer(**optimizer_kwargs)
        
    def loop(self, env, agent, database, n_episodes=10, train=True,
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False, wandb_project=False, 
            save_model=True, save_model_each=50, MODEL_PATH='', 
            save_step_each=1, greedy_sampling=False, initialization=True,
            init_buffer_size=1000):

        min_action = env.action_space.low
        max_action = env.action_space.high

        best_return = -np.infty

        if store_video:
            video = cv2.VideoWriter(video_folder+env.spec.id+'.avi', 0, 40, (500, 500))

        initialized = not (initialization and train)
        returns = []
        for episode in range(0, n_episodes):
            step_counter = 0
            episode_done = False
            state = env.reset()
            episode_return = 0.0

            while not episode_done:
                if initialized:                
                    action = agent.sample_action(state)
                else:
                    action = env.action_space.sample()
                scaled_action = scale_action(action, min_action, max_action).reshape(-1)
                next_state, reward, done, info = env.step(scaled_action)

                if 'task' in state:
                    observation = cat_state_task(state)
                    next_observation = cat_state_task(next_state)
                    step = ExperienceFirstLevel(observation, action, reward, 
                                                done, next_observation)
                elif 'inner_state' in state:
                    step = ExperienceFirstLevel(state['inner_state'], action, reward, 
                                                done, next_state['inner_state'])
                else:
                    raise RuntimeError('Unrecognized state type')

                if store_video:
                    img = env.render('rgb_array')
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                should_save_step = (step_counter % save_step_each) == 0
                if should_save_step:
                    database.append(step)

                episode_return += reward
                state = next_state.copy()

                step_counter += 1

                should_train_in_this_step = train and ((step_counter % train_each) == 0) and initialized 
                if should_train_in_this_step:
                    metrics = self.optimizer.optimize(agent, database)                    
                    if wandb_project and metrics is not None:
                        metrics['step'] = step_counter
                        wandb.log(metrics)

                if step_counter >= max_episode_steps or done:
                    episode_done = True
                
                initialized = initialized or (database.__len__() > init_buffer_size)

            returns.append(episode_return)

            if wandb_project:
                wandb.log({'episode': episode, 'return': episode_return})

            if save_model and ((episode + 1) % save_model_each == 0):
                agent.save(MODEL_PATH)            
                        
            if train and (episode_return > best_return):
                best_return = episode_return
                agent.save(MODEL_PATH + 'best_')

        return_array = np.array(returns)

        if store_video:
            video.release()

        return return_array    


class Second_Level_Trainer:
    def __init__(self, sac_baselines=False, optimizer_kwargs={}):
        if not sac_baselines:
            self.optimizer = Second_Level_SAC_PolicyOptimizer(**optimizer_kwargs)
        else:
            self.optimizer = Second_Level_SAC_with_baselines_PolicyOptimizer(**optimizer_kwargs)
        
    def loop(self, env, agent, database, n_episodes=10, train=True,
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False, wandb_project=False, 
            save_model=True, save_model_each=50, MODEL_PATH='', 
            save_step_each=2, greedy_sampling=False, initialization=True,
            init_buffer_size=500, n_step_td=2):

        best_return = -np.infty

        if store_video:
            video = cv2.VideoWriter(video_folder+env.spec.id+'.mp4', 0x7634706d, 30, (1024, 1024))

        initialized = not (initialization and train)
        returns = []
        for episode in range(0, n_episodes):
            state_buffer = collections.deque(maxlen=n_step_td)
            action_buffer = collections.deque(maxlen=n_step_td)
            reward_buffer = collections.deque(maxlen=n_step_td)

            step_counter = 0
            episode_done = False
            state = env.reset()
            episode_return = 0.0

            state_buffer.append(state)

            while not episode_done:
                if initialized:
                    skill, dist = agent.sample_action(state, explore=(not greedy_sampling))
                else:
                    skill = np.random.randint(agent._n_actions)
                    dist = np.ones(agent._n_actions) / agent._n_actions
                next_state, reward, done, info = self.second_level_step(env, agent, state, skill)

                action_buffer.append(skill)
                reward_buffer.append(reward)
                dist = (dist + 1e-6) / (dist+1e-6).sum()
                entropy = -(dist * np.log(dist)).sum()
                entropy_baseline = self.optimizer.H_mean
                if entropy_baseline is None:
                    entropy_baseline = entropy
                entropy_difference = entropy - entropy_baseline
                alpha = agent.second_level_architecture.get_alpha()
                gamma_n = gamma = self.optimizer.discount_factor
                for previous_step in range(0, len(reward_buffer)-1):
                    reward_buffer[-2-previous_step] += gamma_n * (reward + alpha * entropy_difference)
                    gamma_n *= gamma

                if store_video:
                    img_1 = env.sim.render(width=1024, height=512, depth=False, camera_name='front_camera')[::-1,:,:]
                    img_2 = env.sim.render(width=512, height=512, depth=False, camera_name='global_camera')[::-1,:,:]
                    img_3 = env.sim.render(width=512, height=512, depth=False, camera_name='global_camera_2')[::-1,:,:]
                    #assert img_1.shape == img_2.shape, 'Incompatible dimensions: img1:' + str(img_1.shape) + ', img2:' + str(img_2.shape)
                    img_up = np.concatenate((img_2, img_3), axis=1)
                    img = np.concatenate((img_up, img_1), axis=0)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                buffer_ready = len(state_buffer) == n_step_td
                if buffer_ready:
                    initial_state = state_buffer[0]
                    initial_skill = action_buffer[0]
                    n_step_reward = reward_buffer[0]                    
                    step = PixelExperienceSecondLevel(initial_state['inner_state'], initial_state['outer_state'], 
                                                    initial_skill, n_step_reward, done, next_state['inner_state'], 
                                                    next_state['outer_state'])

                    should_save_step = (step_counter % save_step_each) == 0
                    if should_save_step:
                        database.append(step)

                episode_return += reward
                state = next_state.copy()
                state_buffer.append(state)

                step_counter += 1

                should_train_in_this_step = train and ((step_counter % train_each) == 0) and initialized 
                if should_train_in_this_step:
                    metrics = self.optimizer.optimize(agent, database, n_step_td)                    
                    if wandb_project and metrics is not None:
                        metrics['step'] = step_counter
                        wandb.log(metrics)

                if step_counter >= max_episode_steps or done:
                    episode_done = True
                
                initialized = initialized or (database.__len__() > init_buffer_size)

            returns.append(episode_return)

            if wandb_project:
                wandb.log({'episode': episode, 'return': episode_return})

            if save_model and ((episode + 1) % save_model_each == 0):
                agent.save(MODEL_PATH)
            
            if train and (episode_return > best_return):
                best_return = episode_return
                agent.save(MODEL_PATH + 'best_')

        return_array = np.array(returns)

        if store_video:
            video.release()

        return return_array    

    def second_level_step(self, env, agent, state, skill):
        n_steps = agent._temporal_ratio
        first_level_step_counter = 0
        loop_reward = 0.0
        loop_done = False
        finished_loop = False

        while not finished_loop:
            action = agent.sample_first_level_action(state, skill)
            next_state, reward, done, info = env.step(action)
            loop_reward += reward
            loop_done = loop_done or done
            first_level_step_counter += 1            
            finished_loop = loop_done or ((first_level_step_counter % n_steps) == 0)
            state = next_state.copy()  

        return next_state, loop_reward, loop_done, info            


