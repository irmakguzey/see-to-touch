# Script for overall rewarder
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch

from abc import ABC, abstractmethod

from see_to_touch.utils import get_inverse_image_norm

class Rewarder(ABC):
    def __init__(
        self,
        expert_demos,
        device,
        sinkhorn_rew_scale,
        auto_rew_scale_factor = 10,
        representation_types=['image'],
        expert_frame_matches=-1, 
        episode_frame_matches=-1,
        image_encoder=None,
        tactile_encoder=None,
    ): 
        self.representation_types = representation_types
        self.expert_frame_matches = expert_frame_matches
        self.episode_frame_matches = episode_frame_matches
        self.device = device # This could be in a separate device as well
        self.sinkhorn_rew_scale = sinkhorn_rew_scale
        self.auto_rew_scale_factor = auto_rew_scale_factor
        self.set_expert_demos(expert_demos)
        self.set_encoders(image_encoder, tactile_encoder)

    @abstractmethod
    def get(self, obs): # Method to get the reward
        pass # This should be implemented by each rewarder

    def update_scale(self, current_rewards): 
        sum_rewards = np.sum(current_rewards)
        self.sinkhorn_rew_scale = self.sinkhorn_rew_scale * self.auto_rew_scale_factor / float(np.abs(sum_rewards))

    def get_representations(self, obs, debug=True): 
        # Get the episode representations
        episode_reprs = []
        if 'image' in self.representation_types:
            if self.episode_frame_matches == -1: # It means get all the observations
                image_reprs = self.image_encoder(obs['image_obs'].to(self.device))
            else:
                image_reprs = self.image_encoder(obs['image_obs'][-self.episode_frame_matches:,:].to(self.device))
            episode_reprs.append(image_reprs)
        if 'tactile' in self.representation_types:
            if self.episode_frame_matches == -1:
                tactile_reprs = obs['tactile_repr'].to(self.device)
            else:
                tactile_reprs = obs['tactile_repr'][-self.episode_frame_matches:,:].to(self.device)
            episode_reprs.append(tactile_reprs)
        episode_repr = torch.concat(episode_reprs, dim=-1).detach()   

        # Get the expert representations
        expert_reprs = []
        for expert_id in range(len(self.expert_demos)):
            curr_expert_reprs = []
            if 'image' in self.representation_types:
                if self.expert_frame_matches == -1: # It means get all the observations
                    image_reprs = self.image_encoder(self.expert_demos[expert_id]['image_obs'].to(self.device))
                else:
                    image_reprs = self.image_encoder(self.expert_demos[expert_id]['image_obs'][-self.expert_frame_matches:,:].to(self.device))
                curr_expert_reprs.append(image_reprs)
            if 'tactile' in self.representation_types:
                if self.expert_frame_matches == -1:
                    tactile_reprs = self.expert_demos[expert_id]['tactile_repr'].to(self.device)
                else:
                    tactile_reprs = self.expert_demos[expert_id]['tactile_repr'][-self.expert_frame_matches:,:].to(self.device)
                curr_expert_reprs.append(tactile_reprs)
            curr_expert_repr = torch.concat(curr_expert_reprs, dim=-1).detach()
            expert_reprs.append(curr_expert_repr)
        # expert_reprs = torch.stack(expert_reprs, dim=0)

        # NOTE: Debugging purposes to see the images
        if debug:
            inv_image_transform = get_inverse_image_norm()
            episode_img = inv_image_transform(obs['image_obs'][-1,:]) 
            expert_img = inv_image_transform(self.expert_demos[0]['image_obs'][-1,:]) 

            # Dump the images
            plt.imshow(np.transpose(episode_img.detach().cpu().numpy(), (1,2,0)))
            plt.savefig('episode_img.png')

            plt.imshow(np.transpose(expert_img.detach().cpu().numpy(), (1,2,0)))
            plt.savefig('expert_img.png')

        return episode_repr, expert_reprs # This will be returned in get method

    def set_expert_demos(self, expert_demos): 
        self.expert_demos = expert_demos

    def set_encoders(self, image_encoder=None, tactile_encoder=None):
        self.image_encoder = image_encoder.to(self.device) if not image_encoder is None else None
        self.tactile_encoder = tactile_encoder.to(self.device) if not tactile_encoder is None else None