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

class DexterityEnv(gym.Env):
    def __init__(
        self,
        tactile_out_dir, 
        tactile_model_type = 'byol',
        host_address = '17.24.71.240',
        camera_num = 1,
        height = 480, 
        width = 480, 
        action_type = 'robot',
        data_representations = ['image', 'tactile', 'allegro', 'kinova']
    ):
        self.width = width
        self.height = height
        self.view_num = camera_num 
        self.data_reprs = data_representations

        self.deploy_api = DeployAPI(
            host_address=host_address,
            required_data={'rgb_idxs': [camera_num], 'depth_idxs': []}
        )

        self.set_home_state()
        self._hand = AllegroKDL()
        
        # Get the tactile encoder
        device = torch.device('cuda:0')

        if 'tactile' in data_representations:
            tactile_cfg, tactile_encoder, _ = init_encoder_info(device, tactile_out_dir, 'tactile', model_type=tactile_model_type)
            tactile_img = TactileImage(
                tactile_image_size = tactile_cfg.tactile_image_size, 
                shuffle_type = None
            )
            tactile_repr_dim = tactile_cfg.encoder.tactile_encoder.out_dim if tactile_model_type == 'bc' else tactile_cfg.encoder.out_dim
            self.tactile_repr = TactileRepresentation(
                encoder_out_dim = tactile_repr_dim,
                tactile_encoder = tactile_encoder,
                tactile_image = tactile_img,
                representation_type = 'tdex'
            )

        action_dim = 0
        if 'allegro' in data_representations:
            action_dim += 16
        if 'kinova' in data_representations or 'franka' in data_representations:
            action_dim += 7 
        self.action_type = action_type
        self.action_space = spaces.Box(low = np.array([-1]*action_dim,dtype=np.float32), # Actions are 12 + 7
                                        high = np.array([1]*action_dim,dtype=np.float32),
                                        dtype = np.float32)
        
        
        obs_space_dict = dict()
        if 'image' in data_representations:
            obs_space_dict['pixels'] = spaces.Box(low = np.array([0,0],dtype=np.float32), high = np.array([255,255], dtype=np.float32), dtype = np.float32)
        if 'tactile' in data_representations:
            obs_space_dict['tactile'] = spaces.Box(low = np.array([-1]*tactile_repr_dim, dtype=np.float32),
                                                   high = np.array([1]*tactile_repr_dim, dtype=np.float32),
                                                   dtype = np.float32)
        if 'allegro' in data_representations or 'kinova' in data_representations:
            obs_space_dict['features'] = spaces.Box(low = np.array([-1]*action_dim, dtype=np.float32),
                                                    high = np.array([1]*action_dim, dtype=np.float32),
                                                    dtype = np.float32)
        self.observation_space = spaces.Dict(obs_space_dict)
        
        self.image_subscriber = ZMQCameraSubscriber(
            host = host_address,
            port = 10005 + self.view_num,
            topic_type = 'RGB'
        )

        self.image_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(self._crop_transform),
            T.Resize((self.height, self.width))
        ])

    def set_home_state(self):
        raise NotImplementedError # This method should be implemented by every class that inherits this

    def set_up_env(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

    def _get_curr_image(self):
        image, _ = self.image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image, 'RGB')
        img = self.image_transform(image)
        img = np.asarray(img)
        
        return img # NOTE: This is for environment

    def _crop_transform(self, image):
        return crop_transform(image, camera_view=self.view_num)

    def init_hand(self):
        self.deploy_api.send_robot_action(self.home_state)

    def step(self, action):
        print('action.shape: {}'.format(action.shape))
        try: 
            hand_joint_action = action[:16]
            if self.action_type == 'human': # Then we will not be sending the kinova action 
                self.deploy_api.send_robot_action({
                    'allegro': hand_joint_action
                })
            elif self.action_type == 'robot':
                robot_action_dict = {}
                for data in self.data_reprs:
                    if data == 'allegro':
                        robot_action_dict[data] = hand_joint_action
                    if data == 'franka' or data == 'kinova':
                        robot_action_dict[data] = action[-7:]

                self.deploy_api.send_robot_action(robot_action_dict)
        except:
            print("IK error")
        
        # Get the observations
        obs = self._get_obs()      

        reward, done, infos = 0, False, {'is_success': False} 

        return obs, reward, done, infos 

    def render(self, mode='rbg_array', width=0, height=0):
        return self._get_curr_image()

    def reset(self): 
        self.init_hand()
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        obs = {}
        features_dict = self.deploy_api.get_robot_state()
        feature_arr = []
        if 'allegro' in self.data_reprs:
            feature_arr.append(features_dict[data_name]['position'])

        for data_name in self.data_reprs:
            if data_name == 'image':
                obs['pixels'] = self._get_curr_image() # NOTE: Check this - you're returning non normalized things though
            if data_name == 'tactile':
                sensor_state = self.deploy_api.get_sensor_state()
                tactile_values = sensor_state['xela']['sensor_values']
                with torch.no_grad():
                    obs['tactile'] = self.tactile_repr.get(tactile_values)
            if data_name == 'kinova' or data_name == 'franka': # Arms are added in the end in order to have consistency
                feature_arr.append(features_dict[data_name])

        obs['features'] = np.concatenate(
            feature_arr,
            axis=0
        )

        return obs

    def get_reward():
        pass