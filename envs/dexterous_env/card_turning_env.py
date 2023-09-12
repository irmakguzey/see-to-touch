# Main script for hand interractions 
import numpy as np

from .env import DexterityEnv

class CardTurning(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            kinova = np.array([-0.56546915,  0.28352877,  0.32529727, -0.04745373, -0.74813676, -0.02540322,  0.66135776]), 
            allegro = np.array([
                0, -0.17453293, 0.78539816, 0.78539816,           # Index
                0, -0.17453293,  0.78539816,  0.78539816,         # Middle
                0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
                1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
            ])
        )
