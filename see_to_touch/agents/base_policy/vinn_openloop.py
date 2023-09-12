import numpy as np
from torchvision import transforms as T

from see_to_touch.utils import VISION_IMAGE_MEANS, VISION_IMAGE_STDS
from see_to_touch.models import init_encoder_info

from .base_policy import BasePolicy

# Will choose the first frame demonstration and move from there
class VINNOpenloop(BasePolicy):
    def __init__(
        self,
        expert_demos,
        image_out_dir,
        device,
        image_model_type='byol',
        **kwargs
    ):
        self.device = device
        self.expert_id = 0
        self.set_expert_demos(expert_demos)

        # Load the image encoders
        self.image_encoder, _ = init_encoder_info(
            device = self.device,
            out_dir = image_out_dir,
            encoder_type = 'image',
            model_type = image_model_type
        )

        # Get the first frame expert demonstrations
        self._get_first_frame_exp_representations()

        # Set the image normalization
        self.image_transform = T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)

    def _get_first_frame_exp_representations(self):
        exp_representations = [] 
        for expert_id in range(len(self.expert_demos)):
            first_frame_exp_obs = self.expert_demos[expert_id]['image_obs'][0:1,:].to(self.device) # Expert demos are already normalized
            first_frame_exp_representation = self.image_encoder(first_frame_exp_obs)
            exp_representations.append(first_frame_exp_representation.detach().cpu().numpy().squeeze())

        self.exp_first_frame_reprs = np.stack(exp_representations, axis=0)

    def _get_closest_expert_id(self, obs):
        # Get the representation of the current observation
        image_obs = self.image_transform(obs['image_obs'] / 255.).unsqueeze(0).to(self.device)
        curr_repr = self.image_encoder(image_obs).detach().cpu().numpy().squeeze()

        # Get the distance of the curr_repr to the expert representaitons of the first frame
        l1_distances = self.exp_first_frame_reprs - curr_repr
        l2_distances = np.linalg.norm(l1_distances, axis=1)
        sorted_idxs = np.argsort(l2_distances)

        return sorted_idxs[0], l2_distances
        
    def act(self, obs, episode_step, **kwargs):
        # Get the expert id
        if episode_step == 0:
            self.expert_id, _ = self._get_closest_expert_id(self, obs)

        # Use expert_demos for base action retrieval
        is_done = False
        if episode_step >= len(self.expert_demos[self.expert_id]['actions']):
            episode_step = len(self.expert_demos[self.expert_id]['actions'])-1
            is_done = True

        action = self.expert_demos[self.expert_id]['actions']

        return action, is_done
