# Helper script to load models
import cv2
import glob
import hydra
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T

from pathlib import Path
from PIL import Image as im
from omegaconf import OmegaConf
from tqdm import tqdm 

from holobot.constants import *
from holobot.utils.network import ZMQCameraSubscriber
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from tactile_learning.models import * 
from tactile_learning.tactile_data import *
from tactile_learning.utils import *

from .deployer import Deployer

class BET(Deployer):
    def __init__(
        self,
        data_path,
        deployment_dump_dir,
        seq_length,
        tactile_out_dir=None, # If these are None then it's considered we'll use non trained encoders
        image_out_dir=None,
        image_model_type='byol',
        bet_model_out_dir=None,
        representation_types=['image', 'tactile', 'kinova', 'allegro', 'torque'],
        view_num = 0, # View number to use for image
        demos_to_use=[],
    ):
        
        self.set_up_env() 

        self.representation_types = ['image', 'tactile']
        self.demos_to_use = demos_to_use
        self.representation_types = representation_types
        self.seq_length = seq_length

        self.device = torch.device('cuda:0')
        self.view_num = view_num

        # Load the image and tactile encoders
        tactile_cfg, tactile_encoder, _ = self._init_encoder_info(self.device, tactile_out_dir, 'tactile')
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size, 
            shuffle_type = None
        )
        self.tactile_repr = TactileRepresentation(
            encoder_out_dim = tactile_cfg.encoder.out_dim,
            tactile_encoder = tactile_encoder,
            tactile_image = self.tactile_img,
            representation_type = 'tdex'
        )
        self.image_cfg, self.image_encoder, self.image_transform = init_encoder_info(self.device, image_out_dir, encoder_type='image', model_type=image_model_type)
        self.inv_image_transform = get_inverse_image_norm()
        
        # Load the BET model
        self.bet_model = self._load_bet_model(self.device, bet_model_out_dir)
        self.temporal_representations = None

        self.kdl_solver = AllegroKDL()
        self.state_id = 0

        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data_path = data_path
        self.data = load_data(self.roots, demos_to_use=demos_to_use) # This will return all the desired indices and the values

        self.deployment_dump_dir = deployment_dump_dir
        if not self.deployment_dump_dir is None:
            os.makedirs(self.deployment_dump_dir, exist_ok=True)
    
    def _load_bet_model(self, device, out_dir):
        cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        bet_model = hydra.utils.instantiate(cfg.learner.model).to(self.device)

        model_path = Path(os.path.join(out_dir, 'models/cbet_model.pt'))
        bet_model.load_model(model_path)

        bet_model.eval()
        return bet_model

    def _init_encoder_info(self, device, out_dir, encoder_type='tactile'): # encoder_type: either image or tactile
        if encoder_type == 'tactile' and  out_dir is None:
            encoder = alexnet(pretrained=True, out_dim=512, remove_last_layer=True)
            cfg = OmegaConf.create({'encoder':{'out_dim':512}, 'tactile_image_size':224})
        
        elif encoder_type =='image' and out_dir is None: # Load the pretrained encoder 
            encoder = resnet18(pretrained=True, out_dim=512) # These values are set
            cfg = OmegaConf.create({"encoder":{"out_dim":512}})
        
        else:
            cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
            model_path = os.path.join(out_dir, 'models/byol_encoder_best.pt')
            encoder = load_model(cfg, device, model_path)
        encoder.eval() 
        
        if encoder_type == 'image':
            transform = T.Compose([
                T.Resize((480,640)),
                T.Lambda(self._crop_transform),
                T.Resize(480),
                T.ToTensor(),
                T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
            ]) 
        else:
            transform = None # This is separately set for tactile

        return cfg, encoder, transform
    
    def _crop_transform(self, image):
        return crop_transform(image, camera_view=self.view_num)

    def _load_dataset_image(self, demo_id, image_id):
        dset_img = load_dataset_image(self.data_path, demo_id, image_id, self.view_num)
        img = self.image_transform(dset_img)
        return torch.FloatTensor(img) 
    
    # tactile_values: (N,16,3) - N: Number of sensors
    # robot_states: { allegro: allegro_tip_positions: 12 - 3*4, End effector cartesian position for each finger tip
    #                 kinova: kinova_states : (3,) - Cartesian position of the arm end effector}
    def _get_one_representation(self, image, tactile_values, robot_states):
        for i,repr_type in enumerate(self.representation_types):
            if repr_type == 'allegro' or repr_type == 'kinova' or repr_type == 'torque':
                new_repr = robot_states[repr_type] # These could be received directly from the robot states
            elif repr_type == 'tactile':
                new_repr = self.tactile_repr.get(tactile_values)
            elif repr_type == 'image':
                new_repr = self.image_encoder(image.unsqueeze(dim=0).to(self.device)) # Add a dimension to the first axis so that it could be considered as a batch
                new_repr = new_repr.detach().cpu().numpy().squeeze()

            if i == 0:
                curr_repr = new_repr 
            else: 
                curr_repr = np.concatenate([curr_repr, new_repr], axis=0)
                
        return curr_repr

    def save_deployment(self):
        with open(os.path.join(self.deployment_dump_dir, 'deployment_info.pkl'), 'wb') as f:
            pickle.dump(self.deployment_info, f)

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        # Get the current representation as the observation
        allegro_joint_state = recv_robot_state['allegro']
        fingertip_positions = self.kdl_solver.get_fingertip_coords(allegro_joint_state) # - fingertip position.shape: (12)
        kinova_cart_state = recv_robot_state['kinova']
        allegro_joint_torque = recv_robot_state['torque']
        curr_robot_state = dict(
            allegro = fingertip_positions,
            kinova = kinova_cart_state,
            torque = allegro_joint_torque
        )

        # Get the current visual image
        image = self._get_curr_image()

        # Get the representation with the given tactile value
        curr_repr = self._get_one_representation(
            image,
            tactile_values, 
            curr_robot_state
        ) 
        curr_repr = torch.FloatTensor(curr_repr).unsqueeze(0).unsqueeze(0).to(self.device)

        if self.state_id == 0:
            self.temporal_representations = curr_repr.repeat(1, self.seq_length, 1) # Batch Number, Sequence, Repr Dimension
        else:
            self.temporal_representations = torch.roll(self.temporal_representations, shifts=(0,-1,0), dims=(0,1,2)) # Shift all the old representations to earlier
            self.temporal_representations[-1] = curr_repr # Add the current representation as the last representation

        print('curr_repr.shape: {}'.format(curr_repr.shape))
        print('self.temporal_representations.mean(-1): {}'.format(
            self.temporal_representations.mean(-1)
        ))

        # Sample the action from the model
        action_values, _, _ = self.bet_model(self.temporal_representations, None, None)
        action_values = action_values.detach().cpu().numpy().squeeze()
        print('action_values.shape: {}'.format(
            action_values.shape
        ))

        action = dict(
            # allegro = action_values[0,-1,:16],
            allegro = action_values[-1,:ALLEGRO_JOINT_NUM],
            kinova = action_values[-1,-KINOVA_CARTESIAN_POS_SIZE:] # Apply the last learned action
        )
        print('action: {}'.format(action))
        
        if visualize: 
            image = self._get_curr_image()
            curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
            curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)

            tactile_image = self.tactile_img.get_tactile_image_for_visualization(tactile_values)
            dump_whole_state(tactile_values, tactile_image, None, None, title='curr_state', vision_state=curr_image_cv2)
            curr_state = cv2.imread('curr_state.png')
            image_path = os.path.join(self.deployment_dump_dir, f'state_{str(self.state_id).zfill(2)}.png')
            cv2.imwrite(image_path, curr_state)

        self.state_id += 1
        print(f'STATE ID: {self.state_id}')
        return  action

    def _get_data_with_id(self, id, visualize=False):
        demo_id, tactile_id = self.data['tactile']['indices'][id]
        _, allegro_tip_id = self.data['allegro_tip_states']['indices'][id]
        _, kinova_id = self.data['kinova']['indices'][id]
        _, image_id = self.data['image']['indices'][id]
        _, allegro_state_id = self.data['allegro_joint_states']['indices'][id]

        tactile_value = self.data['tactile']['values'][demo_id][tactile_id] # This should be (N,16,3)
        allegro_tip_position = self.data['allegro_tip_states']['values'][demo_id][allegro_tip_id] # This should be (M*3,)
        kinova_state = self.data['kinova']['values'][demo_id][kinova_id]
        image = self._load_dataset_image(demo_id, image_id)
        
        if visualize:
            tactile_image = self.tactile_img.get_tactile_image_for_visualization(tactile_value) 
            kinova_cart_pos = kinova_state[:3] # Only position is used
            vis_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
            vis_image = cv2.cvtColor(vis_image*255, cv2.COLOR_RGB2BGR)

            visualization_data = dict(
                image = vis_image,
                kinova = kinova_cart_pos, 
                allegro = allegro_tip_position, 
                tactile_values = tactile_value,
                tactile_image = tactile_image
            )
            return visualization_data

        else:
            allegro_joint_torque = self.data['allegro_joint_states']['torques'][demo_id][allegro_state_id] # This is the torque to be used
            robot_states = dict(
                allegro = allegro_tip_position,
                kinova = kinova_state,
                torque = allegro_joint_torque
            )
            data = dict(
                image = image,
                tactile_value = tactile_value, 
                robot_states = robot_states
            )
            return data
