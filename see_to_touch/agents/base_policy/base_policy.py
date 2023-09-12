# Abstract classes for base policy implementations

from abc import ABC, abstractmethod

# Base class for all the base policies we have - will be used in FISH
class BasePolicy(ABC):
    @abstractmethod
    def act(self, obs, episode_step, **kwargs):
        pass

    def set_expert_demos(self, expert_demos): 
        self.expert_demos = expert_demos