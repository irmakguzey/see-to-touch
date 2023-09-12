# Main script for hand interractions 
import cv2 
import gym
import numpy as np
import os
import torch
import torchvision.transforms as T

from gym import spaces
from holobot_api import DeployAPI
from holobot.robot.allegro.allegro_kdl import AllegroKDL
from holobot.utils.network import ZMQCameraSubscriber
from PIL import Image as im

from tactile_learning.tactile_data import TactileImage, TactileRepresentation
from tactile_learning.models import init_encoder_info
from tactile_learning.utils import *

from .env import DexterityEnv

class PlierClipping(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            kinova = np.array( [-0.36715621,  0.32464796,  0.32484341, -0.48690107, -0.5191986 ,
       -0.4718976 ,  0.52026236]), 
            allegro = np.array([
                    -0.054724470713178805, 0.7763095312801784, 1.3082784021851843, 0.4821031579922018, 
                    -0.07396810503219373, 0.7317162161373739, 1.1924368143433348, 0.5210979599511029, 
                    0.09905968916025955, 0.8386652034532028, 1.2098498297443316, 0.5928990060990388, 
                    1.3250477205024265, 0.47479637807926595, 0.5425928229929902, -0.0018403981227994612  # Thumb
            ])
        )