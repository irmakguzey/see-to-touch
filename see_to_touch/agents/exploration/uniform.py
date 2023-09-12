import random

from .explorer import Explorer

class Uniform(Explorer):
    def explore(self, offset_action, global_step):
        if global_step < self.num_expl_steps:
            offset_action.uniform_(-1.0, 1.0)

        return offset_action