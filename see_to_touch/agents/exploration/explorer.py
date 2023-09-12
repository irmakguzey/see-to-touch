# Base module for exploring

from abc import ABC, abstractmethod

class Explorer(ABC): 
    def __init__(
        self,
        num_expl_steps
    ):
        self.num_expl_steps = num_expl_steps
    
    @abstractmethod
    def explore(self, offset_action, global_step, **kwargs):
        pass # This should be implemented by each explorer module