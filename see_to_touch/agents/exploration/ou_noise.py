import numpy as np
import torch

from .explorer import Explorer
from see_to_touch.utils import OrnsteinUhlenbeckActionNoise

class OUNoise(Explorer):
    def __init__(
        self, 
        num_expl_steps,
        sigma=0.8
    ):
        super().__init__(num_expl_steps=num_expl_steps)
        self.sigma = sigma
        self.ou_noise = OrnsteinUhlenbeckActionNoise(
            mu = np.zeros(23), # The mean of the offsets should be 0
            sigma = sigma # It will give bw -1 and 1 - then this gets multiplied by the scale factors ...
        )

    def explore(self, offset_action, global_step, episode_step, device, eval_mode, **kwargs):
        if eval_mode: # If we are evaluating no exploration
            return offset_action
        
        if episode_step == 0:
            self.ou_noise = OrnsteinUhlenbeckActionNoise(
                mu = np.zeros(23),
                sigma = self.sigma
            )
        if global_step < self.num_expl_steps:
            offset_action = torch.FloatTensor(self.ou_noise()).to(device).unsqueeze(0)

        return offset_action