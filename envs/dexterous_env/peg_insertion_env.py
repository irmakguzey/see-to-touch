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

class PegInsertion(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            kinova = np.array([-0.51666868,  0.31732166,  0.29643977, -0.10299692, -0.71015126, -0.03510073,  0.69558954]), 
            allegro = np.array([
            0, -0.17453293, 0.78539816, 0.78539816,           # Index
            0, -0.17453293,  0.78539816,  0.78539816,         # Middle
            0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
            1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
        ])
        )
