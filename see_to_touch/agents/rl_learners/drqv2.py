# TODO

import torch
import torch.nn as nn
import torch.nn.functional as F

from see_to_touch.utils import TruncatedNormal, schedule
from see_to_touch.models import weight_init

from .rl_learner import RLLearner

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim, offset_mask=None):
        super().__init__()


        self.policy = nn.Sequential(nn.Linear(repr_dim + action_shape[0]*100, hidden_dim), # TODO: Why multiply with 100??
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, action_shape[0]))
        # self.offset_mask = torch.tensor(offset_mask).float().to(torch.device('cuda')) # NOTE: This is used to set the exploration

        self.apply(weight_init)

    def forward(self, obs, action, std):

        action = action.repeat(1, 100) # Action shape (1, A) -> (1, 100*A)
        print('action.shape in Actor forward: {}'.format(action.shape))
        h = torch.cat((obs, action), dim=1) # h shape: (1, 100*A + Repr_Dim)
        mu = self.policy(h) 
        # mu = torch.tanh(mu) * self.offset_mask
        mu = torch.tanh(mu)

        # std = torch.ones_like(mu) * std * self.offset_mask
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        # NOTE: This was not in the original fish paper!!!
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
        nn.Linear(feature_dim + action_shape[0], hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
        nn.Linear(feature_dim + action_shape[0], hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(weight_init) # This function already includes orthogonal weight initialization

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2

class DRQv2(RLLearner):
    def __init__(self, action_shape, device,
                 actor, critic, critic_target, lr, critic_target_tau,
                 hand_offset_scale_factor, arm_offset_scale_factor,
                 stddev_schedule, stddev_clip, data_representations, **kwargs):

        super().__init__()

        self.action_shape = action_shape
        self.device = device 

        self.hand_offset_scale_factor = hand_offset_scale_factor
        self.arm_offset_scale_factor = arm_offset_scale_factor

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = critic_target.to(self.device)
        self.critic_target_tau = critic_target_tau

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.stddev_clip = stddev_clip
        self.stddev_schedule = stddev_schedule

        self.data_reprs = data_representations

        self.train()
        self.critic_target.train()

    # def train(self, training=True):
    #     self.training = training
    #     self.actor.train(training)
    #     self.critic.train(training)

    def _scale_action(self, offset_action):
        if 'allegro' in self.data_reprs:
            offset_action[:,:-7] *= self.hand_offset_scale_factor
        if 'franka' in self.data_reprs or 'kinova' in self.data_reprs:
            offset_action[:,-7:] *= self.arm_offset_scale_factor

    def act(self, obs, base_action, global_step, eval_mode, **kwargs):
        stddev = schedule(self.stddev_schedule, global_step)
        dist = self.actor(obs, base_action, stddev)
        action = dist.mean if eval_mode else dist.sample()
        return action
    
    def update_critic(self, obs, action, base_next_action, reward, discount, next_obs, step, **kwargs):
        metrics = dict() 

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, base_next_action, stddev)

            offset_action = dist.sample(clip=self.stddev_clip)
            # offset_action[:,:-7] *= self.hand_offset_scale_factor # NOTE: There is something wrong here?
            # offset_action[:,-7:] *= self.arm_offset_scale_factor 
            self._scale_action(offset_action)
            next_action = base_next_action + offset_action

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        self.critic_opt.step()

        metrics['train_critic/critic_target_q'] = target_Q.mean().item()
        metrics['train_critic/critic_q1'] = Q1.mean().item()
        metrics['train_critic/critic_q2'] = Q2.mean().item()
        metrics['train_critic/loss'] = critic_loss.item()
            
        return metrics
    
    def update_actor(self, obs, base_action, step, **kwargs):
        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)

        dist = self.actor(obs, base_action, stddev)
        action_offset = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action_offset).sum(-1, keepdim=True)

        # action_offset[:,:-7] *= self.hand_offset_scale_factor
        # action_offset[:,-7:] *= self.arm_offset_scale_factor
        self._scale_action(action_offset)

        action = base_action + action_offset 
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()

        self.actor_opt.step()
        
        metrics['train_actor/loss'] = actor_loss.item()
        metrics['train_actor/actor_logprob'] = log_prob.mean().item()
        metrics['train_actor/actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['train_actor/actor_q'] = Q.mean().item()
        metrics['rl_loss'] = -Q.mean().item()

        return metrics
    
    def save_snapshot(self):
        keys_to_save = ['actor', 'critic']
        payload = {k: self.__dict__[k] for k in keys_to_save}