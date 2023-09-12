# Base class for the agent module
# It will set the expert demos and the encoders
import numpy as np 
import torch

from abc import ABC, abstractmethod

from see_to_touch.models import * 
from see_to_touch.utils import * 
from see_to_touch.tactile_data import *

class Agent(ABC):
    def __init__(
        self,
        data_path,
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        data_representations,
        **kwargs
    ):
        
        # Set each given variable to a class variable
        self.__dict__.update(kwargs)

        # Demo based parameters
        self.data_reprs = data_representations
        self._set_data(data_path, expert_demo_nums, data_representations)

        # Get the expert demos and set the encoders
        self._set_image_transform()
        self._set_encoders(
            image_out_dir = image_out_dir, 
            image_model_type = image_model_type,
            tactile_out_dir = tactile_out_dir, 
            tactile_model_type = tactile_model_type
        )
        self._set_expert_demos()

    def _set_data(self, data_path, expert_demo_nums):
        self.data_path = data_path 
        self.expert_demo_nums = expert_demo_nums
        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))

        self.data = load_data(self.roots,
                             demos_to_use=expert_demo_nums,
                             representations=self.data_reprs)
        print('self.data.keys in agent: {}'.format(self.data.keys()))

    def _set_encoders(self, image_out_dir=None, image_model_type=None, tactile_out_dir=None, tactile_model_type=None): 
        if 'image' in self.data_reprs:
            _, self.image_encoder, _  = init_encoder_info(self.device, image_out_dir, 'image', view_num=self.view_num, model_type=image_model_type)
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False 

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

    def train(self, training=True):
        self.training = training
        # TODO: Be careful here!
        # self.actor.train(training) # If these don't exist this should return false
        # self.critic.train(training)

    def _set_image_transform(self):
        self.image_act_transform = T.Compose([
            RandomShiftsAug(pad=4),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        # self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        self.inv_image_transform = get_inverse_image_norm() # This is only to be able to 

    def _set_expert_demos(self): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        # print('IN _SET_EXPERT_DEMOS')
        self.expert_demos = []
        image_obs = [] 
        tactile_reprs = []
        actions = []
        old_demo_id = -1

        ex_key = list(self.data.keys())[0]
        pbar = tqdm(total=len(self.data[ex_key]['indices']))
        for step_id in range(len(self.data[ex_key]['indices'])): 
            # Set observations
            demo_id, _ = self.data[ex_key]['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):
                
                demo_dict = dict()
                if 'image' in self.data_reprs:
                    demo_dict['image_obs'] = torch.stack(image_obs, 0)
                if 'tactile' in self.data_reprs:
                    demo_dict['tactile_repr'] = torch.stack(tactile_reprs, 0)
                if 'allegro' in self.data_reprs or 'kinova' in self.data_reprs or 'franka' in self.data_reprs:
                    demo_dict['actions'] = np.stack(actions, 0)

                self.expert_demos.append(demo_dict)
                image_obs = [] 
                tactile_reprs = []
                actions = []

            
            if 'tactile' in self.data_reprs:
                _, tactile_id = self.data['tactile']['indices'][step_id]
                tactile_value = self.data['tactile']['values'][demo_id][tactile_id]
                tactile_repr = self.tactile_repr.get(tactile_value, detach=False)
                tactile_reprs.append(tactile_repr)

            if 'image' in self.data_reprs:
                _, image_id = self.data['image']['indices'][step_id]
                image = load_dataset_image(
                    data_path = self.data_path, 
                    demo_id = demo_id, 
                    image_id = image_id,
                    view_num = self.view_num,
                    transform = self.image_transform
                )
                image_obs.append(image)

            # Set actions
            action_arr = []
            if 'allegro' in self.data_reprs: 
                _, hand_action_id = self.data['hand_actions']['indices'][step_id]
                hand_action = self.data['hand_actions']['values'][demo_id][hand_action_id]
                action_arr.append(hand_action)
            if 'kinova' in self.data_reprs or 'franka' in self.data_reprs:
                _, arm_id = self.data['arm']['indices'][step_id]
                arm_action = self.data['arm']['values'][demo_id][arm_id]
                action_arr.append(arm_action)

            if len(action_arr) > 0:
                demo_action = np.concatenate(action_arr, axis=-1)
                actions.append(demo_action)

            old_demo_id = demo_id

            pbar.update(1)
            pbar.set_description('Setting the expert demos ')

        pbar.close()

    def _get_policy_reprs_from_obs(self, representation_types, image_obs=None, tactile_repr=None, features=None):
         # Get the representations
        reprs = []
        if 'image' in representation_types:
            # Current representations
            image_obs = self.image_act_transform(image_obs.float()).to(self.device)
            image_reprs = self.image_encoder(image_obs)
            reprs.append(image_reprs)

        if 'tactile' in representation_types:
            tactile_reprs = tactile_repr.to(self.device) # This will give all the representations of one batch
            reprs.append(tactile_reprs)

        if 'features' in representation_types:
            repeated_features = features.repeat(1, self.features_repeat)
            reprs.append(repeated_features.to(self.device))

        return torch.concat(reprs, axis=-1) # Concatenate the representations to get the final representations

    @abstractmethod
    def update(self, **kwargs):
        pass 

    @abstractmethod
    def act(self, **kwargs): 
        pass

    def save_snapshot(self, keys_to_save=['actor']):
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload
    
    def repr_dim(self, type='policy'):
        representations = self.policy_representations if type=='policy' else self.goal_representations
        repr_dim = 0
        if 'tactile' in representations:
            repr_dim += self.tactile_repr.size
        if 'image' in representations:
            repr_dim += 512
        if 'features' in representations:
            if 'allegro' in self.data_reprs:
                repr_dim += 16 * self.features_repeat
            if 'franka' in self.data_reprs or 'kinova' in self.data_reprs:
                repr_dim += 7 * self.features_repeat

        return repr_dim

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
