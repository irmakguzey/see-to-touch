# Base policy for vinn with rollout steps - it finds the step to continue
# end then has that for following few steps

import numpy as np
import torch

from see_to_touch.models import *
from see_to_touch.utils import *
from see_to_touch.tactile_data import *
from see_to_touch.deployers.utils.nn_buffer import NearestNeighborBuffer

from .base_policy import BasePolicy
from .vinn import VINN

# Will choose the first frame demonstration and move from there
class VINNContinuous(VINN):
    def __init__(
        self,
        expert_demos,
        tactile_repr_size,
        image_out_dir,
        device,
        max_steps,
        continuous_steps, # Number of steps to continue
        beginning_demo_cont=False, # Boolean to set to continue from the beginning demo
        image_model_type='byol',
        **kwargs
    ):
        super().__init__( # Call VINN init
            expert_demos, 
            tactile_repr_size,
            image_out_dir,
            device,
            max_steps, 
            image_model_type,
            **kwargs
        )

        self.beginning_demo_cont = beginning_demo_cont
        self.cont_steps = continuous_steps 
        self.steps_from_last_neighbor = 0 # Counter to count the number of steps that we've taken after finding the closest neighbors

    def act(self, obs, episode_step, **kwargs):
        is_done = False
        if self.steps_from_last_neighbor % self.cont_steps == 0:
            
            # Get the current representation
            image_obs = obs['image_obs'].unsqueeze(0) / 255.
            tactile_repr = obs['tactile_repr'].numpy()
            image_obs = self.image_normalize(image_obs.float()).to(self.device)
            with torch.no_grad():
                image_repr = self.image_encoder(image_obs).detach().cpu().numpy().squeeze()
            curr_repr = np.concatenate([image_repr, tactile_repr], axis=0)

            # Choose the action with the buffer 
            _, nn_idxs, _ = self.knn.get_k_nearest_neighbors(curr_repr, k=self.nn_k)
            id_of_nn = self.buffer.choose(nn_idxs)
            nn_id = nn_idxs[id_of_nn]

            # Check if the closest neighbor is the last frame
            curr_demo_id, curr_frame_id = self._turn_id(frame_id=nn_id, frame2expert=True)
            next_demo_id, next_frame_id = self._turn_id(frame_id=nn_id+1, frame2expert=True)
            if next_demo_id != curr_demo_id: # If the predicted state is the last state of the demo
                is_done = True
                print('IS_DONE -> TRUE - curr_demo_id/frame_id: {}/{} next_demo_id/frame_id: {}/{},'.format(
                    curr_demo_id, curr_frame_id, next_demo_id, next_frame_id
                ))
                # next_frame_id -= 1 # Just send the last frame action
            elif episode_step > self.max_steps: 
                is_done = True

            # If beginning_demo_cont is set to true demo won't change
            # after the beginning
            if (self.beginning_demo_cont and episode_step==0) or (not self.beginning_demo_cont):
                self.demo_id = next_demo_id
            self.frame_id = next_frame_id
            
            # Choose a new neighbor
            self.steps_from_last_neighbor = 0 

        else:
            # Continue from the chosen demo
            self.frame_id += 1

            if episode_step > self.max_steps: 
                is_done = True

        # Check if current demo is being finished
        if self.frame_id >= len(self.expert_demos[self.demo_id]['actions']):
            is_done = True
            self.frame_id -= 1
        action = self.expert_demos[self.demo_id]['actions'][self.frame_id]
        self.steps_from_last_neighbor += 1

        print('EPISODE STEP: {}/{}, DEMO ID: {}, FRAME ID: {}, CONT STEPS: {} IS_DONE: {}'.format(
            episode_step, self.max_steps, self.demo_id, self.frame_id, self.cont_steps, is_done
        ))

        if is_done: 
            self.steps_from_last_neighbor = 0
            self.frame_id = 0 # Set the frame_id to 0 automatically

        return action, is_done
        