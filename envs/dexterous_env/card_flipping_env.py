# Main script for hand interractions 
import numpy as np

from .env import DexterityEnv

class CardFlipping(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            kinova = np.array([-0.5707984 ,  0.2903423 ,  0.31730247, -0.0505348 , -0.80014342, -0.02483405,  0.59715998]), 
            allegro = np.array([
                -0.055535682780422854, 0.03907992617559122, 0.5147753186418194, 0.748602214657691,
                -0.10264223897187673, 0.011392517338036723, 0.3647570384028202, 0.773809080963551,
                -0.011080934919122539, 0.06960798594413742, 0.3502775197909466, 0.7669559795594911,
                1.1622167757299338, 0.405469152239451, 0.12007841939021036, 0.6319962825242011
            ])
        )
