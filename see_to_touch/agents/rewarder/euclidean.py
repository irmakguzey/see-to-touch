# Rewarder module for only using cosine simliarity
import numpy as np
import torch 
import sys

from .rewarder import Rewarder

class Euclidean(Rewarder):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def get(self, obs): 
        # Get representations 
        episode_repr, expert_reprs = self.get_representations(obs)

        all_rewards = []
        cost_matrices = []
        best_reward_sum = - sys.maxsize
        for expert_id, expert_repr in enumerate(expert_reprs):
            expert_repr = torch.cat((expert_repr, expert_repr[-1].unsqueeze(0)))
            rewards = -(episode_repr - expert_repr).norm(dim=1)
            rewards *= self.sinkhorn_rew_scale
            rewards = rewards.detach().cpu().numpy()
            
            all_rewards.append(rewards)
            sum_rewards = np.sum(rewards)
            cost_matrices.append(rewards) # Here we can consider cost matrix as the reward itself
            if sum_rewards > best_reward_sum:
                best_reward_sum = sum_rewards 
                best_expert_id = expert_id

        final_reward = all_rewards[best_expert_id]
        final_cost_matrix = cost_matrices[best_expert_id]

        return final_reward, final_cost_matrix, best_expert_id

