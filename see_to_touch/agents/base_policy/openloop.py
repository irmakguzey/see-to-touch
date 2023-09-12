from .base_policy import BasePolicy

class Openloop(BasePolicy):
    def __init__(
        self,
        expert_demos,
        expert_id,
        **kwargs
    ):
        self.expert_id = expert_id
        self.set_expert_demos(expert_demos)

        print('len(expert_demos): {}'.format(len(self.expert_demos)))
        
    def act(self, obs, episode_step, **kwargs):
        # Use expert_demos for base action retrieval
        is_done = False
        if episode_step >= len(self.expert_demos[self.expert_id]['actions']):
            episode_step = len(self.expert_demos[self.expert_id]['actions'])-1
            is_done = True

        action = self.expert_demos[self.expert_id]['actions'][episode_step]

        return action, is_done
