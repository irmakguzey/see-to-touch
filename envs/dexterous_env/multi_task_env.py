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

class MultiTask(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            kinova = np.array([-0.52456534,  0.21932657,  0.37987852,  0.04104819, -0.65619761, -0.02854434,  0.75293094]), 
            allegro = np.array([
                -0.0658244726801581, 0.11152991296986751, 0.036465840916854717, 0.29693057660614736, # Index
                -0.09053422635521813, 0.21657171862672447, -0.17754325611897262, 0.27011271061536507, # Middle
                 0.012094523852233988, 0.11196786731996372, -0.017784060790178313, 0.2670852707825862, # Ring
                0.8499175389966154, 0.3062633015641964, 0.7989875369900138, 0.46722180902731736 # Thumb

            ])
        )