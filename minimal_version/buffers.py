import collections

ExperienceFirstLevel = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])
PixelExperienceSecondLevel = collections.namedtuple(
    'Experience', field_names=['inner_state', 'outer_state', 'action', 'reward', 
                                'done', 'next_inner_state', 'next_outer_state'])

class PixelExperienceBuffer:
    def __init__(self, capacity, level=1):
        self.buffer = collections.deque(maxlen=capacity)
        self._level = level

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        if self._level == 1:
            states, actions, rewards, dones, next_states = \
                zip(*[self.buffer[idx] for idx in indices])

            return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)

        elif self._level == 2:
            inner_states, outer_states, actions, rewards, \
                dones, next_inner_states, next_outer_states = \
                zip(*[self.buffer[idx] for idx in indices])

            return np.array(inner_states), np.array(outer_states), \
                np.array(actions, dtype=np.uint8), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_inner_states), np.array(next_outer_states)

