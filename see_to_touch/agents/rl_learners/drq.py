# TODO

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from see_to_touch.models.utils import weight_init, mlp
from see_to_touch.utils import SquashedNormal, TruncatedNormal, soft_update_params, to_np

from .rl_learner import RLLearner

# Taken from https://github.com/denisyarats/drq
class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, repr_dim, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()
        
        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(repr_dim, hidden_dim,
                         2 * action_shape[0], hidden_depth)


        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, detach_encoder=False):
        # Splits the output of the linear layer into two chunks 
        # (half of the weights are for the mean, other half is for the std)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1) 

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = TruncatedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, repr_dim, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = mlp(repr_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(repr_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DRQ(RLLearner):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, action_shape, device, 
                 actor, critic, critic_target, discount, init_temperature, lr, critic_target_tau,
                 actions_offset, hand_offset_scale_factor, arm_offset_scale_factor, **kwargs):

        super().__init__()

        self.action_shape = action_shape
        self.device = device
        self.discount = discount
        
        self.actions_offset = actions_offset
        self.hand_offset_scale_factor = hand_offset_scale_factor
        self.arm_offset_scale_factor = arm_offset_scale_factor

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = critic_target.to(self.device)
        self.critic_target_tau = critic_target_tau

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
         # Maybe this could be good since it would somehow learn how to explore better
        self.target_entropy = -action_shape[0]

         # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, eval_mode=False, **kwargs): # NOTE: Here it will not matter how the action is calculated because we will be getting the 
        # obs = torch.FloatTensor(obs).to(self.device) # NOTE: These values will be given prepated to input to the actor
        # obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.mean if eval_mode else dist.sample()
        # action = action.clamp(*self.action_range) # This already gets multiplied with the 
        # assert action.ndim == 2 and action.shape[0] == 1
        # return to_np(action)
        return action

    # Higher level agent wrapper will give these values!
    def update_critic(self, obs, obs_aug, action, base_next_action, reward, next_obs,
                      next_obs_aug, **kwargs):
        
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            if self.actions_offset:
                next_offset_action = dist.rsample()
                next_offset_action[:,:-7] *= self.hand_offset_scale_factor
                next_offset_action[:,-7:] *= self.arm_offset_scale_factor
                next_action = base_next_action + next_offset_action
                log_prob = dist.log_prob(next_offset_action).sum(-1, keepdim=True)
            else:
                next_action = dist.rsample()
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (self.discount * target_V) # NOTE: not_done? Should it be added?

            dist_aug = self.actor(next_obs_aug)
            if self.actions_offset:
                next_offset_action_aug = dist_aug.rsample()
                next_offset_action_aug[:,:-7] *= self.hand_offset_scale_factor
                next_offset_action_aug[:,-7:] *= self.arm_offset_scale_factor
                next_action_aug = base_next_action + next_offset_action_aug
                log_prob_aug = dist_aug.log_prob(next_offset_action_aug).sum(-1, keepdim=True)
            else:
                next_action_aug = dist_aug.rsample()
                log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
            
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics['train_critic/loss'] = critic_loss
        return metrics

    def update_actor(self, obs, base_action, **kwargs): # This will also update the alpha as well
        metrics = dict()

        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample() # rsample() can backpropagate whereas sample() cannot
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        if self.actions_offset:
            offset_action = action
            offset_action[:,:-7] = self.hand_offset_scale_factor
            offset_action[:,-7:] = self.arm_offset_scale_factor
            action = base_action + offset_action

        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Update the metrics 
        metrics['train_actor/loss'] = actor_loss
        metrics['train_actor/target_entropy'] = self.target_entropy
        metrics['train_actor/entropy'] = -log_prob.mean()
        metrics['train_alpha/loss'] = alpha_loss 
        metrics['train_alpha/value'] = self.alpha

        return metrics

    def update_critic_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def save_snapshot(self):
        keys_to_save = ['actor', 'critic', 'log_alpha']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def load_snapshot_eval(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v

        self.critic_target.load_state_dict(self.critic.state_dict())

    # def update(self, obs, action, base_action, reward, discount, next_obs, base_next_action):
    #     metrics = dict() 

    #     if step % self.update_every_steps != 0:
    #         return metrics 
        
    #     batch = next(replay_iter)
        

    # def update(self, replay_buffer, logger, step):
    #     obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
    #         self.batch_size)

    #     logger.log('train/batch_reward', reward.mean(), step)

    #     self.update_critic(obs, obs_aug, action, reward, next_obs,
    #                        next_obs_aug, not_done, logger, step)

    #     if step % self.actor_update_frequency == 0:
    #         self.update_actor_and_alpha(obs, logger, step)

    #     if step % self.critic_target_update_frequency == 0:
    #         soft_update_params(self.critic, self.critic_target, self.critic_tau)