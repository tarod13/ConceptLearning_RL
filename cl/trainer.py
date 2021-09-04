import collections
import numpy as np
from buffers import ExperienceFirstLevel, PixelExperienceSecondLevel
from policy_optimizers import Second_Level_SAC_PolicyOptimizer

import wandb

import cv2
video_folder = '/home/researcher/Diego/Concept_Learning_minimal/videos/'

class Trainer:
    def __init__(self, train_level=2, optimizer_kwargs={}):
        self._train_level = train_level
        self.optimizer = Second_Level_SAC_PolicyOptimizer(**optimizer_kwargs)

    def loop(self, env, agent, database, n_episodes=10, train=True,
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False, wandb_project=False, 
            save_model=True, save_model_each=50, MODEL_PATH='', 
            save_step_each=2, greedy_sampling=False, initialization=True,
            init_buffer_size=500):

        if store_video:
            video = cv2.VideoWriter(video_folder+env.spec.id+'.avi', 0, 40, (256, 256+128))

        initialized = not initialization
        returns = []
        for episode in range(0, n_episodes):
            step_counter = 0
            episode_done = False
            state = env.reset()
            episode_return = 0.0

            while not episode_done:

                if self._train_level == 1:
                    action = agent.sample_action(state)
                    next_state, reward, done, info = env.step(action)
                    step = ExperienceFirstLevel(state['inner_state'], action, reward, 
                                                done, next_state['inner_state'])

                elif self._train_level == 2:
                    if initialized:
                        skill = agent.sample_action(state, explore=(not greedy_sampling))
                    else:
                        skill = np.random.randint(agent._n_actions)
                    next_state, reward, done, info = self.second_level_step(env, agent, state, skill)
                    step = PixelExperienceSecondLevel(state['inner_state'], state['outer_state'], 
                                                    skill, reward, done, next_state['inner_state'], 
                                                    next_state['outer_state'])                    
                
                if store_video:
                    img_1 = env.sim.render(width=256, height=128, depth=False, camera_name='front_camera')[::-1,:,:]
                    img_2 = env.sim.render(width=256, height=256, depth=False, camera_name='global_camera')[::-1,:,:]
                    #assert img_1.shape == img_2.shape, 'Incompatible dimensions: img1:' + str(img_1.shape) + ', img2:' + str(img_2.shape)
                    img = np.concatenate((img_1, img_2), axis=0)
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


