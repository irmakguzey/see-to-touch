# Rewarder module for only using cosine simliarity
import numpy as np
import torch 
import sys

from see_to_touch.utils import get_inverse_image_norm, structural_similarity_index

from .rewarder import Rewarder

class SSIM(Rewarder):
    def __init__(
            self,
            ssim_base_factor,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ssim_base_factor = ssim_base_factor
        self.inv_image_transform = get_inverse_image_norm()

    def get(self, obs): 
        # Get representations 
        # episode_repr, expert_reprs = self.get_representations(obs)

        all_rewards = []
        cost_matrices = []
        best_reward_sum = - sys.maxsize
        for expert_id in range(len(self.expert_demos)):
            
            if self.episode_frame_matches == -1:
                episode_img = self.inv_image_transform(obs['image_obs'])
            else:
                episode_img = self.inv_image_transform(obs['image_obs'][-self.episode_frame_matches:,:])

            if self.expert_frame_matches == -1:
                expert_img = self.inv_image_transform(self.expert_demos[expert_id]['image_obs'])
            else: 
                expert_img = self.inv_image_transform(self.expert_demos[expert_id]['image_obs'][-self.expert_frame_matches:,:])
            
            rewards = structural_similarity_index(
                x = expert_img,
                y = episode_img,
                base_factor = self.ssim_base_factor
            )
            rewards *= self.sinkhorn_rew_scale
            cost_matrix = torch.FloatTensor(rewards)
            
            all_rewards.append(rewards)
            sum_rewards = np.sum(rewards)
            cost_matrices.append(cost_matrix) # Here we can consider cost matrix as the reward itself
            if sum_rewards > best_reward_sum:
                best_reward_sum = sum_rewards 
                best_expert_id = expert_id

        final_reward = all_rewards[best_expert_id]
        final_cost_matrix = cost_matrices[best_expert_id]

        return final_reward, final_cost_matrix, best_expert_id

