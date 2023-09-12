# Base module for exploring

import torch

from abc import ABC, abstractmethod
from tactile_learning.utils import soft_update_params

class RLLearner(ABC): 

    @abstractmethod
    def act(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_actor(self, **kwargs):
        raise NotImplementedError 

    @abstractmethod
    def update_critic(self, **kwargs):
        raise NotImplementedError  
    
    def update_critic_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def save_snapshot(self, **kwargs):
        keys_to_save = ['actor', 'critic']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload
    
    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def load_snapshot_eval(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v

        self.critic_target.load_state_dict(self.critic.state_dict())
