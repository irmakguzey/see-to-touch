# Script to deploy already created demo
import glob

from holobot.constants import *

from .deployer import Deployer
from tactile_learning.utils import load_data

class OpenLoop(Deployer):
    def __init__(
        self,
        data_path, # root in string
        data_representations,
        demo_to_run,
        apply_hand_states = False, # boolean to indicate if we should apply commanded allegro states or actual allegro states
        deployment_dump_dir = None
    ):

        # roots = glob.glob(f'{data_path}/demonstration_*')
        # roots = sorted(roots)
        # self.data = load_data(roots, demos_to_use=[demo_to_run])
        super().__init__(data_path=data_path, data_representations=data_representations)
        self._set_data(demos_to_use=[demo_to_run])

        print('self.data.keys(): {}'.format(self.data.keys()))

        self.state_id = 0
        self.hand_action_key = 'hand_joint_states' if apply_hand_states else 'hand_actions'

    def get_action(self, **kwargs):

        action = dict()

        if 'allegro' in self.data_reprs:
            demo_id, action_id = self.data[self.hand_action_key]['indices'][self.state_id] 
            hand_action = self.data[self.hand_action_key]['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)

        if 'franka' in self.data_reprs or 'kinova' in self.data_reprs:
            demo_id, arm_id = self.data['arm']['indices'][self.state_id] 
            arm_action = self.data['arm']['values'][demo_id][arm_id] # Get the next saved kinova_state

        for data in self.data_reprs:
            if data == 'allegro':
                action[data] = hand_action
            if data == 'franka' or data == 'kinova':
                action[data] = arm_action

        self.state_id += 1

        return action

    def save_deployment(self): # We don't really need to do anything here
        pass