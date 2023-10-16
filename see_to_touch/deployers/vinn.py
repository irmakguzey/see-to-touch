# Helper script to load models
import cv2
import numpy as np
import os
import pickle
import torch

from tqdm import tqdm 

from holobot.constants import *
from holobot.robot.allegro.allegro_kdl import AllegroKDL

from see_to_touch.models import *
from see_to_touch.tactile_data import *
from see_to_touch.utils import *

from .deployer import Deployer
from .utils.nn_buffer import NearestNeighborBuffer

class VINN(Deployer):
    def __init__(
        self,
        data_path,
        data_representations,
        deployment_dump_dir,
        tactile_out_dir=None, # If these are None then it's considered we'll use non trained encoders
        tactile_model_type = 'byol',
        image_out_dir=None,
        image_model_type = 'byol',
        representation_types = ['image', 'tactile', 'kinova', 'allegro', 'torque'], # Torque could be used
        representation_importance = [1,1,1,1], 
        tactile_repr_type = 'tdex', # raw, shared, stacked, tdex, sumpool, pca (uses the encoder passed)
        tactile_shuffle_type = None,   
        nn_buffer_size=100,
        nn_k=20,
        demos_to_use=[0],
        view_num = 0, # View number to use for image
        open_loop = False, # Open loop vinn means that we'll run the demo after getting the first frame from KNN
        dump_deployment_info = False
    ):

        super().__init__(
            data_path = data_path, 
            data_representations = data_representations
        )

        self.set_up_env() 

        self.representation_types = representation_types
        self.demos_to_use = demos_to_use

        self.device = torch.device('cuda:0')
        self.view_num = view_num
        self.open_loop = open_loop

        self._set_encoders(
            image_out_dir = image_out_dir, 
            image_model_type = image_model_type,
            tactile_out_dir = tactile_out_dir,
            tactile_model_type = tactile_model_type
        )

        self._set_data(demos_to_use)

        self._get_all_representations()
        self.state_id = 0 # Increase it with each get_action

        self.nn_k = nn_k
        self.kdl_solver = AllegroKDL()
        self.buffer = NearestNeighborBuffer(nn_buffer_size)
        self.knn = ScaledKNearestNeighbors(
            self.all_representations, # Both the input and the output of the nearest neighbors are
            self.all_representations,
            representation_types,
            representation_importance,
            tactile_repr_size = self.tactile_repr.size if 'tactile' in self.data_reprs else None
        )

        self.deployment_dump_dir = deployment_dump_dir
        if dump_deployment_info:
            os.makedirs(self.deployment_dump_dir, exist_ok=True)
        self.dump_deployment_info = dump_deployment_info
        if dump_deployment_info:
            self.deployment_info = dict(
                all_representations = self.all_representations,
                curr_representations = [], # representations will be appended to this list
                closest_representations = [],
                neighbor_ids = [],
                images = [], 
                tactile_values = []
            )

        self.visualization_info = dict(
            state_ids = [],
            images = [], 
            tactile_values = [],
            id_of_nns = [], 
            nn_idxs = [], 
            nn_dists = []
        )

    def _crop_transform(self, image):
        return crop_transform(image, camera_view=self.view_num)

    def _load_dataset_image(self, demo_id, image_id):
        dset_img = load_dataset_image(self.data_path, demo_id, image_id, self.view_num)
        img = self.image_transform(dset_img)
        return torch.FloatTensor(img) 
    
    # tactile_values: (N,16,3) - N: Number of sensors
    # robot_states: { allegro: allegro_tip_positions: 12 - 3*4, End effector cartesian position for each finger tip
    #                 kinova: kinova_states : (3,) - Cartesian position of the arm end effector}
    def _get_one_representation(self, repr_data):
        #  image, tactile_values, robot_states
        for i,repr_type in enumerate(self.representation_types):
            if repr_type == 'allegro' or repr_type == 'kinova' or repr_type == 'franka' or repr_type == 'torque':
                new_repr = repr_data['robot_states'][repr_type] # These could be received directly from the robot states
            elif repr_type == 'tactile':
                new_repr = self.tactile_repr.get(repr_data['tactile_value'])
            elif repr_type == 'image':
                # print('repr_data[image].unsqueeze(dim=0).to(self.device) device: {}'.format(
                #     repr_data['image'].unsqueeze(dim=0).to(self.device).get_device()
                # ))
                # print('image encoder device: {}'.format(self.image_encoder))
                new_repr = self.image_encoder(repr_data['image'].unsqueeze(dim=0).to(self.device)) # Add a dimension to the first axis so that it could be considered as a batch
                new_repr = new_repr.detach().cpu().numpy().squeeze()

            if i == 0:
                curr_repr = new_repr 
            else: 
                curr_repr = np.concatenate([curr_repr, new_repr], axis=0)
                
        return curr_repr
    
    def _get_all_representations(self):
        print('Getting all representations')

        self.all_representations = []

        print('self.data.keys(): {}'.format(self.data.keys()))
        ex_key = list(self.data.keys())[0]
        pbar = tqdm(total=len(self.data[ex_key]['indices']))
        for index in range(len(self.data[ex_key]['indices'])):
            # Get the representation data
            repr_data = self._get_data_with_id(index, visualize=False)

            representation = self._get_one_representation(
                repr_data
            )
            self.all_representations.append(representation)
            pbar.update(1)

        pbar.close()

        self.all_representations = np.stack(self.all_representations, axis=0)

    def save_deployment(self):
        if self.dump_deployment_info:
            with open(os.path.join(self.deployment_dump_dir, 'deployment_info.pkl'), 'wb') as f:
                pickle.dump(self.deployment_info, f)

        with open(os.path.join(self.deployment_dump_dir, 'visualization_info.pkl'), 'wb') as f:
            pickle.dump(self.visualization_info, f)

        # Visualize
        if len(self.visualization_info['state_ids']) > 0: # It was asked to visualize
            pbar = tqdm(total=len(self.visualization_info['state_ids']))
            for i,state_id in enumerate(self.visualization_info['state_ids']):

                # print('')
                self._visualize_state(
                    curr_tactile_values = self.visualization_info['tactile_values'][i],
                    curr_fingertip_position = None, # We are not dumping state for now
                    curr_kinova_cart_pos = None,
                    id_of_nn = self.visualization_info['id_of_nns'][i],
                    nn_idxs = self.visualization_info['nn_idxs'][i],
                    nn_separate_dists = self.visualization_info['nn_dists'][i],
                    image = self.visualization_info['images'][i], 
                    state_id = state_id
                )
                pbar.set_description(f'Dumping visualization state: {state_id}')
                pbar.update(1)

            pbar.close()

    def get_action(self, state_dict, visualize=False):
        if self.open_loop:
            if self.state_id == 0: # Get the closest nearest neighbor id for the first state
                action, self.open_loop_start_id = self._get_knn_action(state_dict, visualize)
            else:
                action = self._get_open_loop_action(state_dict, visualize)
        else:
            action = self._get_knn_action(state_dict, visualize)
        
        print(f'STATE ID: {self.state_id}')
        return  action
    
    def _get_action_dict_from_data(self, data_id):
        action = dict()

        if 'allegro' in self.data_reprs:
            demo_id, action_id = self.data['hand_actions']['indices'][data_id] 
            hand_action = self.data['hand_actions']['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)

        if 'franka' in self.data_reprs or 'kinova' in self.data_reprs:
            demo_id, arm_id = self.data['arm']['indices'][data_id] 
            arm_action = self.data['arm']['values'][demo_id][arm_id] # Get the next saved kinova_state

        for data in self.data_reprs:
            if data == 'allegro':
                action[data] = hand_action
            if data == 'franka' or data == 'kinova':
                action[data] = arm_action

        return action
    
    def _get_open_loop_action(self, state_dict, visualize):

        action = self._get_action_dict_from_data(self.state_id + self.open_loop_start_id)

        if visualize: 
            image = self._get_curr_image()
            curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
            curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)

            if 'tactile' in self.data_reprs:
                tactile_values = state_dict['tactile']
                tactile_image = self._get_tactile_image_for_visualization(state_dict['tactile'])
            else:
                tactile_values = None 
                tactile_image = None
            dump_whole_state(vision_state=curr_image_cv2,
                             hand_tip = None,
                             arm_tip = None,
                             tactile_values = tactile_values,
                             tactile_image = tactile_image, 
                             title = 'curr_state')
                
            curr_state = cv2.imread('curr_state.png')
            image_path = os.path.join(self.deployment_dump_dir, f'state_{str(self.state_id).zfill(2)}.png')
            cv2.imwrite(image_path, curr_state)

        self.state_id += 1
        
        return action 
    
    # tactile_values.shape: (16,15,3)
    # robot_state: {allegro: allegro_joint_state (16,), kinova: kinova_cart_state (3,)}
    def _get_knn_action(self, state_dict, visualize=False):
        # Get the current state of the robot
        repr_data = dict(
            robot_states = dict()
        )
        for data in self.data_reprs:
            if data == 'allegro':
                hand_joint_state = state_dict[data]
                hand_tip_pos = self.kdl_solver.get_fingertip_coords(hand_joint_state)
                repr_data['robot_states'][data] = hand_tip_pos 

                hand_joint_torque = state_dict['torque']
                repr_data['robot_states']['torque'] = hand_joint_torque 

            if data == 'kinova' or data == 'franka': # TODO: These checks should be modularized in the end! 
                repr_data['robot_states'][data] = state_dict[data]

            if data == 'image':
                repr_data[data] = self._get_curr_image()

            if data == 'tactile':
                repr_data['tactile_value'] = state_dict['tactile']

        # Get the representation with the given data
        curr_representation = self._get_one_representation(repr_data)

        # Choose the action with the buffer 
        _, nn_idxs, nn_separate_dists = self.knn.get_k_nearest_neighbors(curr_representation, k=self.nn_k)
        id_of_nn = self.buffer.choose(nn_idxs)
        nn_id = nn_idxs[id_of_nn]
        ex_key = list(self.data.keys())[0]
        if nn_id+1 >= len(self.data[ex_key]['indices']): # If the chosen action is the action after the last action
            nn_idxs = np.delete(nn_idxs, id_of_nn)
            id_of_nn = self.buffer.choose(nn_idxs)
            nn_id = nn_idxs[id_of_nn]

        # Get the action dictionary
        nn_action = self._get_action_dict_from_data(nn_id+1)

        # Save everything to deployment_info
        if self.dump_deployment_info:
            self.deployment_info['curr_representations'].append(curr_representation)
            self.deployment_info['images'].append(state_dict['image'])
            self.deployment_info['tactile_values'].append(state_dict['tactile_value'])
            self.deployment_info['neighbor_ids'].append(nn_idxs[0])
            closest_representation = self.all_representations[nn_idxs[0]]
            self.deployment_info['closest_representations'].append(closest_representation)

        # Visualize if given 
        if visualize: 
            self.visualization_info['state_ids'].append(self.state_id)
            self.visualization_info['images'].append(state_dict['image'])
            self.visualization_info['tactile_values'].append(state_dict['tactile_value'])
            self.visualization_info['id_of_nns'].append(id_of_nn)
            self.visualization_info['nn_idxs'].append(nn_idxs)
            self.visualization_info['nn_dists'].append(nn_separate_dists)

        self.state_id += 1

        if self.open_loop:
            return nn_action, nn_id

        return nn_action

    def _visualize_state(self, curr_tactile_values, curr_fingertip_position, curr_kinova_cart_pos, id_of_nn, nn_idxs, nn_separate_dists, state_id=None, image=None):
        # Get the current image 
        if image is None:
            image = self._get_curr_image()
        curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)
        curr_tactile_image = self.tactile_img.get_tactile_image_for_visualization(curr_tactile_values) 

        nn_id = nn_idxs[id_of_nn]
        # Get the next visualization data
        knn_vis_data = self._get_data_with_id(nn_id, visualize=True)

        # Get the demo ids of the closest 3 neighbor
        demo_ids = []
        demo_nums = []
        viz_id_of_nns = []
        for i in range(3):
            if not (id_of_nn+i >= len(nn_idxs)):
                viz_nn_id = nn_idxs[id_of_nn+i]
                viz_id_of_nns.append(id_of_nn+i)
            else:
                viz_id_of_nns.append(viz_id_of_nns[-1])
            demo_id, _ = self.data['tactile']['indices'][viz_nn_id]
            demo_ids.append(demo_id)
            demo_nums.append(int(self.roots[demo_id].split('/')[-1].split('_')[-1]))
    
        dump_whole_state(curr_tactile_values, curr_tactile_image, curr_fingertip_position, curr_kinova_cart_pos, title='curr_state', vision_state=curr_image_cv2)
        dump_whole_state(knn_vis_data['tactile_values'], knn_vis_data['tactile_image'], knn_vis_data['allegro'], knn_vis_data['kinova'], title='knn_state', vision_state=knn_vis_data['image'])
        dump_repr_effects(nn_separate_dists, viz_id_of_nns, demo_nums, self.representation_types)
        if state_id is None:
            state_id = self.state_id
        dump_knn_state(
            dump_dir = self.deployment_dump_dir,
            img_name = 'state_{}.png'.format(str(state_id).zfill(2)),
            image_repr = True,
            add_repr_effects = True,
            include_temporal_states = False
        )

    def _get_data_with_id(self, id, visualize=False):

        data_dict = dict(
            robot_states = dict()
        )
        hand_tip_position = None
        arm_state = None 
        hand_joint_torque = None
        tactile_value = None

        demo_id, _ = self.data[list(self.data.keys())[0]]['indices'][id]
        for data in self.data_reprs:
            
            if data == 'tactile':
                _, tactile_id = self.data['tactile']['indices'][id]
                tactile_value = self.data[data]['values'][demo_id][tactile_id]
                data_dict['tactile_value'] = tactile_value 

            if data == 'allegro':
                _, hand_tip_id = self.data['hand_tip_states']['indices'][id]
                hand_state_id = self.data['hand_joint_states']['indices'][id]
                hand_tip_position = self.data['hand_tip_states']['values'][demo_id][hand_tip_id]
                hand_joint_torque = self.data['hand_joint_states']['torques'][demo_id][hand_state_id]

                data_dict['robot_states'][data] = hand_tip_position 
                data_dict['robot_states']['torque'] = hand_joint_torque

            if data == 'kinova' or data == 'franka':
                _, arm_id = self.data['arm']['indices'][id]
                arm_state = self.data['arm']['values'][demo_id][arm_id]

                data_dict['robot_states'][data] = arm_state

            if data == 'image': 
                _, image_id = self.data['image']['indices'][id]
                image = self._load_dataset_image(demo_id, image_id)

                data_dict['image'] = image

        if visualize: # TODO: This could give error - ignore this for now
            tactile_image = self.tactile_img.get_tactile_image_for_visualization(tactile_value) 
            kinova_cart_pos = arm_state[:3] # Only position is used
            vis_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
            vis_image = cv2.cvtColor(vis_image*255, cv2.COLOR_RGB2BGR)

            visualization_data = dict(
                image = vis_image,
                kinova = kinova_cart_pos, 
                allegro = hand_tip_position, 
                tactile_values = tactile_value,
                tactile_image = tactile_image
            )
            return visualization_data
        else:
            return data_dict