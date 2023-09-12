import hydra
import random
import torch
import torch.utils.data as data 

from omegaconf import DictConfig

# Script to return dataloaders
def get_dataloaders(cfg : DictConfig):
    if 'demobased' in cfg.dataset and cfg.dataset.demobased:
        return get_demobased_shuffled_dataloader(cfg)
    else:
        return get_framebased_shuffled_dataloader(cfg)

    return None, None, None

def get_framebased_shuffled_dataloader(cfg):
    if 'tactile' in cfg.learner_type:
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            data_path = cfg.data_dir,
            tactile_img_size = cfg.tactile_image_size
        ) # This should be named this way 
    else:
        # print()
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            data_path = cfg.data_dir
        )

        # print('cfg.data_dir: {}, dataset: {}'.format(cfg.data_dir, dataset))
        

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(dataset, 
                                             [train_dset_size, test_dset_size],
                                             generator=torch.Generator().manual_seed(cfg.seed))
    train_sampler = data.DistributedSampler(train_dset, drop_last=True, shuffle=True) if cfg.distributed else None
    test_sampler = data.DistributedSampler(test_dset, drop_last=True, shuffle=False) if cfg.distributed else None # val will not be shuffled

    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                                    num_workers=cfg.num_workers, sampler=train_sampler)
    test_loader = data.DataLoader(test_dset, batch_size=cfg.batch_size, shuffle=test_sampler is None,
                                    num_workers=cfg.num_workers, sampler=test_sampler)

    return train_loader, test_loader, dataset

def get_demobased_shuffled_dataloader(cfg):
    random.seed(10)

    all_demos_to_use = cfg.dataset.demos_to_use
    # Get a random demo numbers to shuffle
    test_demo = random.choice(all_demos_to_use)
    test_dset = hydra.utils.instantiate(
        cfg.dataset.dataset,
        data_path = cfg.data_dir,
        demos_to_use = [test_demo],
        dset_type = 'test'
    )
    all_demos_to_use.remove(test_demo)
    train_demos = all_demos_to_use
    train_dset = hydra.utils.instantiate(
        cfg.dataset.dataset,
        data_path = cfg.data_dir,
        demos_to_use=train_demos,
        dset_type = 'train'
    )

    train_sampler = data.DistributedSampler(train_dset, drop_last=True, shuffle=True) if cfg.distributed else None
    test_sampler = data.DistributedSampler(test_dset, drop_last=True, shuffle=False) if cfg.distributed else None # val will not be shuffled

    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                                    num_workers=cfg.num_workers, sampler=train_sampler)
    test_loader = data.DataLoader(test_dset, batch_size=cfg.batch_size, shuffle=test_sampler is None,
                                    num_workers=cfg.num_workers, sampler=test_sampler)

    return train_loader, test_loader, None
    