
import numpy as np

from .env import DexterityEnv

class MintOpening(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            kinova = np.array([-0.56115389,  0.31624374,  0.31479663, -0.07334466, -0.77505636, -0.03475262,  0.62665802]), 
            allegro = np.array([
            0.007651593318878476, 0.27805874641647516, 0.49409139803226626, 0.12588787585164785,
            -0.11940214584593481, 0.19662629377894955, 0.5199197897356908, 0.06314802003100513,
            0.009800427120003513, 0.12339007191971564, 0.48730888519866017, 0.1442982664591681,
            1.2516348347765962, 0.1655223256916746, 0.6311178279922637, 0.022541493408132687
        ])
        )
