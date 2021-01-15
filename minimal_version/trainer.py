import collections
import numpy as np
from buffers import ExperienceFirstLevel, PixelExperienceSecondLevel
from policy_optimizers import Second_Level_SAC_PolicyOptimizer

import wandb

import cv2
video_folder = '/home/researcher/Diego/Concept_Learning_minimal/videos/'

class Trainer:
    def __init__(self, train_level=2):
        self._train_level = train_level
        self.optimizer = Second_Level_SAC_PolicyOptimizer()

    def loop(self, env, agent, database, n_episodes=10, 
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False, wandb_project=False, 
            save_model=True, save_model_each=50, MODEL_PATH=''):

        if store_video:
            video = cv2.VideoWriter(video_folder+env.spec.id+'.avi', 0, 40, (168,84))

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
                    skill = agent.sample_action(state)
                    next_state, reward, done, info = self.second_level_step(env, agent, state, skill)
                    step = PixelExperienceSecondLevel(state['inner_state'], state['outer_state'], 
                                                    skill, reward, done, next_state['inner_state'], 
                                                    next_state['outer_state'])                    
                
                if store_video:
                    img = (np.swapaxes(np.swapaxes(state['outer_state'], 0, 2), 0, 1) * 255.0).astype(np.uint8)
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                database.append(step)

                episode_return += reward
                state = next_state.copy()

                step_counter += 1

                should_train_in_this_step = (step_counter % train_each) == 0 
                if should_train_in_this_step:
                    metrics = self.optimizer.optimize(agent, database)                    
                    if wandb_project and metrics is not None:
                        metrics['episode'] = episode
                        metrics['step'] = step_counter
                        wandb.log(metrics)

                if step_counter >= max_episode_steps or done:
                    episode_done = True
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


