# Base class for the agent module
# It will set the expert demos and the encoders
import numpy as np 
import torch

from abc import ABC, abstractmethod

from see_to_touch.models import * 
from see_to_touch.utils import * 
from see_to_touch.tactile_data import *

class MultitaskAgent(ABC):
    def __init__(
        self,
        data_path,
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        # view_num, device,
        # update_every_steps, update_critic_target_every_steps, update_actor_every_steps, 
        # features_repeat, 
        # experiment_name, # Learning based parts
        **kwargs
    ):
        
        # Set each given variable to a class variable
        self.__dict__.update(kwargs)

        # Demo based parameters
        self._set_data(data_path, expert_demo_nums)

        # self.device = device
        # self.view_num = view_num
        # self.experiment_name = experiment_name

        # # Learning based parameters
        # if not update_critic_target_every_steps is None and not update_actor_every_steps is None:
        #     self.update_every_steps = min(update_actor_every_steps, update_critic_target_every_steps)
            
        # self.features_repeat = features_repeat

        # Get the expert demos and set the encoders
        self._set_image_transform()
        self._set_encoders(
            image_out_dir = image_out_dir, 
            image_model_type = image_model_type,
            tactile_out_dir = tactile_out_dir, 
            tactile_model_type = tactile_model_type
        )

        # For each task, there is an expert demo to set
        self.expert_demos = []
        for task_num in range(len(data_path)):
            self._set_expert_demos(task_num)
            self.expert_demos.append(self.task_demos)
        

    def _set_data(self, data_path, expert_demo_nums):
        #NOTE: data_path, expert_demo_nums, roots, data are all lists
        self.data_path = data_path 
        self.expert_demo_nums = expert_demo_nums

        #NOTE: This part is changed
        self.roots = []
        for task_path in self.data_path: 
            self.roots.append(sorted(glob.glob(f'{task_path}/demonstration_*')))
        #NOTE: Here the self.data is a list of expert demos, each task only one demo is used
        self.data = []
        task_num = 0
        for demo_num in self.expert_demo_nums: 
            self.data.append(load_data(self.roots[task_num], demos_to_use=demo_num))
            task_num += 1 

    def _set_encoders(self, image_out_dir, image_model_type, tactile_out_dir, tactile_model_type): 
        #NOTE: self.image_encoder, self.image_transform need be list 
        self.image_encoder = []
        self.image_transform = []

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

        #NOTE: for image encoders
        for task_num in range(len(image_out_dir)):
            _, task_image_encoder, task_image_transform  = init_encoder_info(self.device, image_out_dir[task_num], 'image', view_num=self.view_num, model_type=image_model_type)

            # Freeze the encoders
            task_image_encoder.eval()
            for param in task_image_encoder.parameters():
                param.requires_grad = False 

            self.image_encoder.append(task_image_encoder)
            self.image_transform.append(task_image_transform)

    def train(self, training=True):
        self.training = training
        # self.actor.train(training) # If these don't exist this should return false
        # self.critic.train(training)

    def _set_image_transform(self):
        self.image_act_transform = T.Compose([
            RandomShiftsAug(pad=4),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        # self.image_normalize = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        self.inv_image_transform = get_inverse_image_norm() # This is only to be able to 




    def _set_expert_demos(self, task_num): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        # print('IN _SET_EXPERT_DEMOS')
        self.task_demos = []
        image_obs = [] 
        tactile_reprs = []
        actions = []
        old_demo_id = -1
        pbar = tqdm(total=len(self.data[task_num]['image']['indices']))
        for step_id in range(len(self.data[task_num]['image']['indices'])): 
            # Set observations
            demo_id, tactile_id = self.data[task_num]['tactile']['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data[task_num]['image']['indices'])-1):
                
                self.task_demos.append(dict(
                    image_obs = torch.stack(image_obs, 0),
                    tactile_repr = torch.stack(tactile_reprs, 0),
                    actions = np.stack(actions, 0)
                ))
                image_obs = [] 
                tactile_reprs = []
                actions = []

            tactile_value = self.data[task_num]['tactile']['values'][demo_id][tactile_id]
            tactile_repr = self.tactile_repr.get(tactile_value, detach=False)

            _, image_id = self.data[task_num]['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path[task_num], 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = self.view_num, #here we assume the view_num will be the same
                transform = self.image_transform[task_num]
            )

            # Set actions
            _, allegro_action_id = self.data[task_num]['allegro_actions']['indices'][step_id]
            allegro_action = self.data[task_num]['allegro_actions']['values'][demo_id][allegro_action_id]
            _, kinova_id = self.data[task_num]['kinova']['indices'][step_id]
            kinova_action = self.data[task_num]['kinova']['values'][demo_id][kinova_id]
            demo_action = np.concatenate([allegro_action, kinova_action], axis=-1)

            image_obs.append(image)
            tactile_reprs.append(tactile_repr)
            actions.append(demo_action)

            old_demo_id = demo_id

            pbar.update(1)
            pbar.set_description('Setting the expert demos ')

        pbar.close()





    def _get_policy_reprs_from_obs(self, image_obs, tactile_repr, features, representation_types, task_num):
         # Get the representations
        reprs = []
        if 'image' in representation_types:
            # Current representations
            image_obs = self.image_act_transform(image_obs.float()).to(self.device)
            image_reprs = self.image_encoder[task_num](image_obs)
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
            repr_dim += 23 * self.features_repeat

        return repr_dim

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v