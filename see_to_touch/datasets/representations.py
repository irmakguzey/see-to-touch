# Dataset that receives saved representations

import glob
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T 

from torch.utils import data
from torchvision.datasets.folder import default_loader as loader 

from tactile_learning.utils import load_data

class SequentialRepresentationsActions(data.Dataset):
    def __init__(
        self,
        seq_length,
        data_path,
        demos_to_use,
        dset_type='all'
    ):
        super().__init__()
        self.seq_length = seq_length

        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=demos_to_use)

        # Make sure that there is a file named all_representations.pkl in the
        # data_path, if not this class should complain
        all_representations_path = os.path.join(data_path, f'{dset_type}_representations.pkl')
        assert os.path.exists(all_representations_path), f'{dset_type}_representations.pkl should exist first, run python preprocess.py repr_preprocessor.apply=true'

        with open(all_representations_path, 'rb') as f:
            self.all_representations = pickle.load(f)

    def __len__(self):
        return len(self.all_representations)-self.seq_length
    
    def _get_actions(self, index):
        def _get_action(curr_id):
            demo_id, allegro_action_id = self.data['allegro_actions']['indices'][curr_id]
            allegro_action = self.data['allegro_actions']['values'][demo_id][allegro_action_id]

            _, kinova_id = self.data['kinova']['indices'][curr_id]
            kinova_action = self.data['kinova']['values'][demo_id][kinova_id+1]

            total_action = np.concatenate([allegro_action, kinova_action], axis=-1)
            return torch.FloatTensor(total_action)
        
        for seq_id in range(self.seq_length):
            if seq_id == 0:
                actions = _get_action(index+seq_id).unsqueeze(0)
            
            else:
                actions = torch.concat([
                    actions,
                    _get_action(index+seq_id).unsqueeze(0)
                ], dim=0)

        return actions
    
    def __getitem__(self, index):
        obs = self.all_representations[index:index+self.seq_length,:]

        act = self._get_actions(index)
        
        return torch.FloatTensor(obs), act
    
if __name__ == '__main__':
    dset = SequentialRepresentationsActions(
        seq_length=3,
        data_path = '/home/irmak/Workspace/Holo-Bot/extracted_data/bowl_picking/after_rss',
        demos_to_use=[22,24,26,34,28,29],
        dset_type='all'
    ) 
    print(len(dset))
    dataloader = data.DataLoader(dset, 
                                batch_size  = 16, 
                                shuffle     = True, 
                                num_workers = 8,
                                pin_memory  = True)

    batch = next(iter(dataloader))
    print('batch[0].shape: {}, batch[1].shape: {}'.format(
        batch[0].shape, batch[1].shape # it should be 16 + 7 (for each joint)
    ))


