import os 
import torch

from abc import ABC, abstractmethod
from PIL import Image as im

from holobot.utils.network import ZMQCameraSubscriber
from see_to_touch.utils import *
from see_to_touch.tactile_data import *
from see_to_touch.models import init_encoder_info

# Base class for all deployment modules
class Deployer(ABC):
    def __init__(self, data_path, data_representations):
        self.data_path = data_path
        self.data_reprs = data_representations

    def set_up_env(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29506"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

    @abstractmethod
    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        pass

    @abstractmethod
    def save_deployment(self):
        pass 

    def _set_data(self, demos_to_use):
        self.roots = sorted(glob.glob(f'{self.data_path}/demonstration_*'))
        self.data = load_data(
            roots = self.roots, 
            demos_to_use = demos_to_use,
            representations = self.data_reprs
        )

        print('self.data[arm][indices]: {}, self.data_reprs: {}'.format(
            self.data['arm']['indices'], self.data_reprs
            ))

    def _set_encoders(self, image_out_dir=None, image_model_type=None, tactile_out_dir=None, tactile_model_type=None):
        if 'image' in self.data_reprs:
            _, self.image_encoder, self.image_transform  = init_encoder_info(self.device, image_out_dir, 'image', view_num=self.view_num, model_type=image_model_type)
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False 

            self.inv_image_transform = get_inverse_image_norm()

        if 'tactile' in self.data_reprs:
            tactile_cfg, self.tactile_encoder, _ = init_encoder_info(self.device, tactile_out_dir, 'tactile', view_num=self.view_num, model_type=tactile_model_type)
            tactile_img = TactileImage(
                tactile_image_size = tactile_cfg.tactile_image_size, 
                shuffle_type = None
            )
            tactile_repr_dim = tactile_cfg.encoder.tactile_encoder.out_dim if tactile_model_type == 'bc' else tactile_cfg.encoder.out_dim
            
            self.tactile_repr = TactileRepresentation( # This will be used when calculating the reward - not getting the observations
                encoder_out_dim = tactile_repr_dim,
                tactile_encoder = self.tactile_encoder,
                tactile_image = tactile_img,
                representation_type = 'tdex',
                device = self.device
            )

            self.tactile_encoder.eval()
            
            for param in self.tactile_encoder.parameters():
                param.requires_grad = False

    def _get_curr_image(self, host='172.24.71.240', port=10005):
        image_subscriber = ZMQCameraSubscriber( # TODO: Change this such that it will create a subscriber only once
            host = host,
            port = port + self.view_num,
            topic_type = 'RGB'
        )
        image, _ = image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image)
        img = self.image_transform(image)
        return torch.FloatTensor(img)