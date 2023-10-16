# Alexnet means and stds
TACTILE_IMAGE_MEANS = [0.485, 0.456, 0.406]
TACTILE_IMAGE_STDS = [0.229, 0.224, 0.225]

PLAY_DATA_PATH = '/data/tactile_learning/play_data' # NOTE: Should download the tactile_learning play data for this

# These constants are used to clamp 
TACTILE_PLAY_DATA_CLAMP_MIN = -1000
TACTILE_PLAY_DATA_CLAMP_MAX = 1000

# Constans for camera images
VISION_IMAGE_MEANS = [0.4191, 0.4445, 0.4409]
VISION_IMAGE_STDS = [0.2108, 0.1882, 0.1835]

# Robotic constants
KINOVA_CARTESIAN_POS_SIZE = 7 
ALLEGRO_JOINT_NUM = 16
ALLEGRO_EE_REPR_SIZE = 12 

# Constants for the IBC training
THUMB_JOINT_LIMITS = [
    [0,1.5], [-0.1,1], [-0.1,1.3], [-0.1,1.3]
]
FINGER_JOINT_LIMITS = [
    [-0.2, 0.2],[-0.1, 1.3],[-0.1,1.3],[-0.1,1.3]
]
KINOVA_JOINT_LIMITS = [
    [-0.48, -0.55],[0.15,0.25], [0.3,0.45], # X,Y,Z
    [-0.1,0.1], [-0.7,-0.5],[0.0,0.1],[0.6,0.8] 
]

PREPROCESS_MODALITY_LOAD_NAMES = { # TODO Should change these with different modalities
    'tactile': 'touch_sensor_values.h5',
    'hand': ['_joint_states.h5', '_commanded_joint_states.h5'],
    'arm': '_cartesian_states.h5',
    'image': 'rgb_video.metadata'
}

PREPROCESS_MODALITY_DUMP_NAMES = {
    'tactile': 'tactile_indices.pkl',
    'arm': '_indices.pkl',
    'hand': ['_indices.pkl', '_action_indices.pkl'],
    'image': 'image_indices.pkl'
}

MODALITY_TYPES = {
    'allegro': 'hand',
    'kinova': 'arm',
    'franka': 'arm',
    'image': 'image',
    'tactile': 'tactile' # TODO: This could be changed to reskin / xela
}