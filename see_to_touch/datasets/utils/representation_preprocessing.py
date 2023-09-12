# Script to turn all the given dataset into representations

# Helper script to load models
import cv2
import glob
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T

from PIL import Image as im
from omegaconf import OmegaConf
from tqdm import tqdm 

from holobot.constants import *

from tactile_learning.tactile_data import *
from tactile_learning.utils import *
from tactile_learning.models import *

# Class that will take all the representations and dump them to the given data directory

class RepresentationPreprocessor: # It should only take image and tactile inout and move accordingly 
    def __init__(
        self,
        data_path, # It will dump the representations to the given data path
        tactile_out_dir=None, # If these are None then it's considered we'll use non trained encoders
        image_out_dir=None,
        representation_types = ['image', 'tactile', 'kinova', 'allegro', 'torque'],
        view_num = 0, # View number to use for image
        demos_to_use = [],
        image_model_type = 'byol'
    ):
        
        self.device = torch.device('cuda:0')
        self.set_up_env()

        self.view_num = view_num
        self.representation_types = representation_types

        tactile_cfg, tactile_encoder, _ = self._init_encoder_info(self.device, tactile_out_dir, 'tactile')
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_cfg.tactile_image_size
        )
        self.tactile_repr = TactileRepresentation(
            encoder_out_dim = tactile_cfg.encoder.out_dim,
            tactile_encoder = tactile_encoder,
            tactile_image = self.tactile_img,
            representation_type = 'tdex'
        )

        self.image_cfg, self.image_encoder, self.image_transform = init_encoder_info(
            device = self.device,
            out_dir = image_out_dir,
            encoder_type = 'image',
            view_num = view_num,
            model_type = image_model_type
        )
        # self.image_cfg, self.image_encoder, self.image_transform = self._init_encoder_info(self.device, image_out_dir, 'image')
        self.inv_image_transform = get_inverse_image_norm()

        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data_path = data_path
        self.data = load_data(self.roots, demos_to_use=demos_to_use) # This will return all the desired indices and the values

    def set_up_env(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29506"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(self.device)
    
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
    
    def get_all_representations(self):
        print('Getting all representations')
        repr_dim = 0
        if 'tactile' in self.representation_types: repr_dim += self.tactile_repr.size
        if 'allegro' in self.representation_types:  repr_dim += ALLEGRO_EE_REPR_SIZE
        if 'kinova' in self.representation_types: repr_dim += KINOVA_CARTESIAN_POS_SIZE
        if 'torque' in self.representation_types: repr_dim += ALLEGRO_JOINT_NUM # There are 16 joint values
        if 'image' in self.representation_types: repr_dim += 512

        print('repr_dim: {}, self.repr_rtypes: {}'.format(
            repr_dim, self.representation_types
        ))

        self.all_representations = np.zeros((
            len(self.data['tactile']['indices']), repr_dim
        ))

        pbar = tqdm(total=len(self.data['tactile']['indices']))
        for index in range(len(self.data['tactile']['indices'])):
            # Get the representation data
            repr_data = self._get_data_with_id(index)

            representation = self._get_one_representation(
                repr_data['image'],
                repr_data['tactile_value'],
                repr_data['robot_states']
            )
            self.all_representations[index, :] = representation[:]
            pbar.update(1)

        pbar.close()

    def dump_all_representations(self, file_name='all_representations.pkl'):
        dumping_file_path = os.path.join(self.data_path, file_name)
        with open(dumping_file_path, 'wb') as f:
            pickle.dump(self.all_representations, f)

    def _get_data_with_id(self, id):
        demo_id, tactile_id = self.data['tactile']['indices'][id]
        _, allegro_tip_id = self.data['allegro_tip_states']['indices'][id]
        _, kinova_id = self.data['kinova']['indices'][id]
        _, image_id = self.data['image']['indices'][id]
        _, allegro_state_id = self.data['allegro_joint_states']['indices'][id]

        tactile_value = self.data['tactile']['values'][demo_id][tactile_id] # This should be (N,16,3)
        allegro_tip_position = self.data['allegro_tip_states']['values'][demo_id][allegro_tip_id] # This should be (M*3,)
        kinova_state = self.data['kinova']['values'][demo_id][kinova_id]
        image = self._load_dataset_image(demo_id, image_id)

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
    
if __name__ == '__main__':
    # preprocessor = RepresentationPreprocessor(
    #     data_path='/home/irmak/Workspace/Holo-Bot/extracted_data/cup_picking/after_rss',
    #     tactile_out_dir='/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
    #     image_out_dir='/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.04.05/00-59_image_byol_bs_32_cup_picking_after_rss',
    #     view_num=1,
    #     demos_to_use=[17], #[13,14,15,16,17,18],
    #     representation_types=['image','tactile','kinova','allegro']
    # )
    # preprocessor.get_all_representations()
    # print(f'Dumping - all_representations.shape: {preprocessor.all_representations.shape}')
    # preprocessor.dump_all_representations(file_name='test_representations.pkl')

    preprocessor = RepresentationPreprocessor(
        data_path='/home/irmak/Workspace/Holo-Bot/extracted_data/cup_picking/after_rss',
        tactile_out_dir='/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.01.28/12-32_tactile_byol_bs_512_tactile_play_data_alexnet_pretrained_duration_120',
        image_out_dir='/home/irmak/Workspace/tactile-learning/tactile_learning/out/2023.04.05/00-59_image_byol_bs_32_cup_picking_after_rss',
        view_num=1,
        demos_to_use=[13,14,15,16,18],
        representation_types=['image','tactile','kinova','allegro']
    )
    preprocessor.get_all_representations()
    print(f'Dumping - all_representations.shape: {preprocessor.all_representations.shape}')
    preprocessor.dump_all_representations(file_name='train_representations.pkl')

