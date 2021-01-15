import collections
import numpy as np
from buffers import ExperienceFirstLevel, PixelExperienceSecondLevel
from policy_optimizers import Second_Level_SAC_PolicyOptimizer

import cv2
video_folder = '/home/researcher/Diego/Concept_Learning_minimal/videos/'

class Trainer:
    def __init__(self, train_level=2):
        self._train_level = train_level
        self.optimizer = Second_Level_SAC_PolicyOptimizer()

    def loop(self, env, agent, database, n_episodes=10, 
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False):

        if store_video:
            video = cv2.VideoWriter(video_folder+env.spec.id+'.avi', 0, 40, (168,84))

        returns = []
        for episode in range(0, n_episodes):
            step_counter = 0
            episode_done = False
            state = env.reset()
            episode_return = 0.0
            while not episode_done:
                a = agent.sample_action(state)
                if self._train_level == 1:
                    action = a
                elif self._train_level == 2:
                    skill, action = a

                next_state, reward, done, info = env.step(action)

                if self._train_level == 1:
                    step = ExperienceFirstLevel(state['inner_state'], action, reward, 
                                                done, next_state['inner_state'])
                elif self._train_level == 2:
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
                    self.optimizer.optimize(agent, database)

                if step_counter >= max_episode_steps or done:
                    episode_done = True
            returns.append(episode_return)
        return_array = np.array(returns)

        if store_video:
            video.release()

        return return_array