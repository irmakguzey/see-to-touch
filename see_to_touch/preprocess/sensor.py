import os 
import cv2
import numpy as np
import pickle 
import h5py
import shutil

from tqdm import tqdm

from .prep_module import PreprocessorModule

class TouchPreprocessor(PreprocessorModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.load_file_name = 'touch_sensor_values.h5'
        self.dump_file_name = 'tactile_indices.pkl'
        self.indices = []

    def __repr__(self):
        return 'touch_reprocessor'

    def load_data(self):
        file_path = os.path.join(self.root, self.load_file_name)
        with h5py.File(file_path, 'r') as f:
            tactile_timestamps = f['timestamps'][()]

        self.data = dict(
            timestamps = tactile_timestamps
        )

    def get_next_timestamp(self):
        return -1 # Tactile is not considered a 'selective' module at all

    