import random
import numpy as np

from .explorer import Explorer

from see_to_touch.utils import exponential_epsilon_decay

class ExponentialExploration(Explorer):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.last_offset = np.zeros(23)

    def explore(self, offset_action, global_step, episode_step):
        epsilon = exponential_epsilon_decay(
            step_idx=global_step,
            epsilon_decay=self.num_expl_steps
        )
        rand_num = random.random()
        if rand_num < epsilon:
            offset_action.uniform_(-1.0, 1.0)
            offset_action = offset_action / ((episode_step+1)/2.) # TODO: This should be tested!
            if episode_step > 0: 
                offset_action += self.last_offset
            self.last_offset = offset_action

        return offset_action