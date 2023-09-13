import numpy as np

from .env import DexterityEnv

class MoveFranka(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            franka = np.array([0.50974673,  0.16168165,  0.34757033,  0.63821834,  0.29931313, -0.30522093,  0.64025694]), 
        )
