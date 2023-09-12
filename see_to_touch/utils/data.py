import glob
import h5py
import numpy as np
import os
import pickle
import torch

from torchvision.datasets.folder import default_loader as loader
from tqdm import tqdm

from .constants import MODALITY_TYPES, PREPROCESS_MODALITY_DUMP_NAMES, PREPROCESS_MODALITY_LOAD_NAMES

# Load human data 
def load_human_data(roots, demos_to_use=[], duration=120):
    roots = sorted(roots)

    keypoint_indices = []
    image_indices = []
    hand_keypoints = {}

    for demo_id,root in enumerate(roots): 
        demo_num = int(root.split('/')[-1].split('_')[-1])
        if (len(demos_to_use) > 0 and demo_num in demos_to_use) or (len(demos_to_use) == 0): # If it's empty then it will be ignored
            with open(os.path.join(root, 'keypoint_indices.pkl'), 'rb') as f:
                keypoint_indices += pickle.load(f)
            with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
                image_indices += pickle.load(f)

            # Load the data
            with h5py.File(os.path.join(root, 'keypoints.h5'), 'r') as f:
                hand_keypoints[demo_id] = f['transformed_hand_coords'][()] 

    # Find the total lengths now
    whole_length = len(keypoint_indices)
    desired_len = int((duration / 120) * whole_length)

    data = dict(
        keypoint = dict(
            indices = keypoint_indices[:desired_len],
            values = hand_keypoints
        ),
        image = dict( 
            indices = image_indices[:desired_len]
        )
    )

    return data

# Method to load all the data from given roots and return arrays for it
def load_data(roots, demos_to_use=[], duration=120, representations=['image','tactile','allegro','kinova']): # If the total length is equal to 2 hrs - it means we want the whole data
    roots = sorted(roots)

    arm_indices = []
    arm_states = {}

    hand_indices = []
    hand_action_indices = []
    hand_tip_positions = {}
    hand_joint_positions = {}
    hand_joint_torques = {}
    hand_actions = {}

    tactile_indices = []
    tactile_values = {} 

    image_indices = []

    repr_types = []
    for demo_id, root in enumerate(roots):
        demo_num = int(root.split('/')[-1].split('_')[-1])
        # print('demo_num: {}'.format(demo_num))
        if (len(demos_to_use) > 0 and demo_num in demos_to_use) or (len(demos_to_use) == 0):
            for repr_name in representations:
                repr_type = MODALITY_TYPES[repr_name]
                repr_types.append(repr_type)

                if repr_type == 'arm':
                    with open(os.path.join(root, f'{repr_name}_indices.pkl'), 'rb') as f:
                        arm_indices += pickle.load(f)
                    with h5py.File(os.path.join(root, f'{repr_name}_cartesian_states.h5'), 'r') as f:
                        state = np.concatenate([f['positions'][()], f['orientations'][()]], axis=1)     
                        arm_states[demo_id] = state
                    print('arm_indices {} in {}'.format(arm_indices, repr_name))

                if repr_type == 'hand':
                    with open(os.path.join(root, f'{repr_name}_indices.pkl'), 'rb') as f:
                        hand_indices += pickle.load(f)
                    with open(os.path.join(root, f'{repr_name}_action_indices.pkl'), 'rb') as f:
                        hand_action_indices += pickle.load(f)
                    with h5py.File(os.path.join(root, f'{repr_name}_fingertip_states.h5'), 'r') as f:
                        hand_tip_positions[demo_id] = f['positions'][()]
                    with h5py.File(os.path.join(root, f'{repr_name}_joint_states.h5'), 'r') as f:
                        hand_joint_positions[demo_id] = f['positions'][()]
                        hand_joint_torques[demo_id] = f['efforts'][()]
                    with h5py.File(os.path.join(root, f'{repr_name}_commanded_joint_states.h5'), 'r') as f:
                        hand_actions[demo_id] = f['positions'][()] # Positions are to be learned - since this is a position control
                
                if repr_type == 'tactile':
                    with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
                        tactile_indices += pickle.load(f)
                    with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
                        tactile_values[demo_id] = f['sensor_values'][()]

                if repr_type == 'image':
                    with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
                        image_indices += pickle.load(f)

    print('REPR TYPES IN LOAD_DATA: {}'.format(
        repr_types
    ))

    # Find the total lengths now
    # whole_length = len(tactile_indices)
    whole_length = max([len(data_idx) for data_idx in [tactile_indices, hand_indices, arm_indices, image_indices]])
    desired_len = int((duration / 120) * whole_length)

    data = dict()
    # for repr_type in repr_types:
    if 'hand' in repr_types:
        data['hand_joint_states'] = dict(
            indices = hand_indices[:desired_len],
            values = hand_joint_positions,
            torques = hand_joint_torques
        )
        data['hand_tip_states'] = dict(
            indices = hand_indices[:desired_len],
            values = hand_tip_positions
        )
        data['hand_actions'] = dict(
            indices = hand_action_indices[:desired_len],
            values = hand_actions
        )
    if 'arm' in repr_types:
        data['arm'] = dict(
            indices = arm_indices[:desired_len], 
            values = arm_states
        )

    if 'tactile' in repr_types:
        data['tactile'] = dict(
            indices = tactile_indices[:desired_len],
            values = tactile_values
        )

    if 'image' in repr_types:
        data['image'] = dict(
            indices = image_indices[:desired_len]
        )

    return data 

    # tactile_indices = [] 
    # allegro_indices = []
    # allegro_action_indices = [] 
    # kinova_indices = []
    # image_indices = []
    
    # tactile_values = {}
    # allegro_tip_positions = {} 
    # allegro_joint_positions = {}
    # allegro_joint_torques = {}
    # allegro_actions = {}
    # kinova_states = {}

    # for demo_id,root in enumerate(roots): 
    #     demo_num = int(root.split('/')[-1].split('_')[-1])
    #     if (len(demos_to_use) > 0 and demo_num in demos_to_use) or (len(demos_to_use) == 0): # If it's empty then it will be ignored
    #         with open(os.path.join(root, 'tactile_indices.pkl'), 'rb') as f:
    #             tactile_indices += pickle.load(f)
    #         with open(os.path.join(root, 'allegro_indices.pkl'), 'rb') as f:
    #             allegro_indices += pickle.load(f)
    #         with open(os.path.join(root, 'allegro_action_indices.pkl'), 'rb') as f:
    #             allegro_action_indices += pickle.load(f)
    #         with open(os.path.join(root, 'kinova_indices.pkl'), 'rb') as f:
    #             kinova_indices += pickle.load(f)
    #         with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
    #             image_indices += pickle.load(f)

    #         # Load the data
    #         with h5py.File(os.path.join(root, 'allegro_fingertip_states.h5'), 'r') as f:
    #             allegro_tip_positions[demo_id] = f['positions'][()]
    #         with h5py.File(os.path.join(root, 'allegro_joint_states.h5'), 'r') as f:
    #             allegro_joint_positions[demo_id] = f['positions'][()]
    #             allegro_joint_torques[demo_id] = f['efforts'][()]
    #         with h5py.File(os.path.join(root, 'allegro_commanded_joint_states.h5'), 'r') as f:
    #             allegro_actions[demo_id] = f['positions'][()] # Positions are to be learned - since this is a position control
    #         with h5py.File(os.path.join(root, 'touch_sensor_values.h5'), 'r') as f:
    #             tactile_values[demo_id] = f['sensor_values'][()]
    #         with h5py.File(os.path.join(root, 'kinova_cartesian_states.h5'), 'r') as f:
    #             state = np.concatenate([f['positions'][()], f['orientations'][()]], axis=1)     
    #             kinova_states[demo_id] = state

    # # Find the total lengths now
    # whole_length = len(tactile_indices)
    # desired_len = int((duration / 120) * whole_length)

    # data = dict(
    #     tactile = dict(
    #         indices = tactile_indices[:desired_len],
    #         values = tactile_values
    #     ),
    #     allegro_joint_states = dict(
    #         indices = allegro_indices[:desired_len], 
    #         values = allegro_joint_positions,
    #         torques = allegro_joint_torques
    #     ),
    #     allegro_tip_states = dict(
    #         indices = allegro_indices[:desired_len], 
    #         values = allegro_tip_positions
    #     ),
    #     allegro_actions = dict(
    #         indices = allegro_action_indices[:desired_len],
    #         values = allegro_actions
    #     ),
    #     kinova = dict( 
    #         indices = kinova_indices[:desired_len], 
    #         values = kinova_states
    #     ), 
    #     image = dict( 
    #         indices = image_indices[:desired_len]
    #     )
    # )

    # return data


def get_image_stats(len_image_dataset, image_loader):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum    += inputs.sum(axis = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    # pixel count
    count = len_image_dataset * 480 * 480

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

def load_dataset_image(data_path, demo_id, image_id, view_num, transform=None, as_int=False):
    roots = glob.glob(f'{data_path}/demonstration_*')
    roots = sorted(roots)
    image_root = roots[demo_id]
    image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(view_num, str(image_id).zfill(5)))
    img = loader(image_path)
    if not transform is None:
        img = transform(img)
        img = torch.FloatTensor(img)
    return img

# Taken from https://github.com/NYU-robot-learning/multimodal-action-anticipation/utils/__init__.py#L90
def batch_indexing(input, idx):
    """
    Given an input with shape (*batch_shape, k, *value_shape),
    and an index with shape (*batch_shape) with values in [0, k),
    index the input on the k dimension.
    Returns: (*batch_shape, *value_shape)
    """
    batch_shape = idx.shape
    dim = len(idx.shape)
    value_shape = input.shape[dim + 1 :]
    N = batch_shape.numel()
    assert input.shape[:dim] == batch_shape, "Input batch shape must match index shape"
    assert len(value_shape) > 0, "No values left after indexing"

    # flatten the batch shape
    input_flat = input.reshape(N, *input.shape[dim:])
    idx_flat = idx.reshape(N)
    result = input_flat[np.arange(N), idx_flat]
    return result.reshape(*batch_shape, *value_shape) 