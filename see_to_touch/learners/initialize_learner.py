import hydra 

from torch.nn.parallel import DistributedDataParallel as DDP

from .byol import BYOLLearner
from .bet import BETLearner
from .temporal_ssl import TemporalSSLLearner
from .behavior_cloning import ImageTactileBC

from see_to_touch.utils import *
from see_to_touch.models import  *

def init_learner(cfg, device, rank=0):
    if 'tactile' in cfg.learner_type:
        return init_tactile_byol(
            cfg,
            device, 
            rank,
            aug_stat_multiplier=cfg.learner.aug_stat_multiplier,
            byol_in_channels=cfg.learner.byol_in_channels,
            byol_hidden_layer=cfg.learner.byol_hidden_layer
        )
    elif cfg.learner_type == 'image_byol':
        return init_image_byol(cfg, device, rank)
    elif cfg.learner_type == 'bet':
        return init_bet_learner(
            cfg,
            device
        )
    elif cfg.learner_type == 'temporal_ssl':
        return init_temporal_learner(
            cfg,
            device,
            rank
        )

    return None



def init_bet_learner(cfg, device):
    bet_model = hydra.utils.instantiate(cfg.learner.model).to(device)

    optimizer = bet_model.configure_optimizers(
        weight_decay=cfg.learner.optim.weight_decay,
        learning_rate=cfg.learner.optim.lr,
        betas=cfg.learner.optim.betas
    )

    learner = BETLearner(
        bet_model = bet_model,
        optimizer = optimizer
    )
    learner.to(device)

    return learner

def init_tactile_byol(cfg, device, rank, aug_stat_multiplier=1, byol_in_channels=3, byol_hidden_layer=-2):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS*aug_stat_multiplier,
        img_stds = TACTILE_IMAGE_STDS*aug_stat_multiplier,
        img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.tactile_image_size,
        augment_fn = augment_fn,
        hidden_layer = byol_hidden_layer,
        in_channels = byol_in_channels
    ).to(device)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'tactile'
    )

    learner.to(device)

    return learner

def init_image_byol(cfg, device, rank):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_vision_augmentations(
        img_means = VISION_IMAGE_MEANS,
        img_stds = VISION_IMAGE_STDS
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.vision_image_size,
        augment_fn = augment_fn
    ).to(device)
    if cfg.distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'image'
    )

    learner.to(device)

    return learner

def init_temporal_learner(cfg, device, rank):
    encoder = hydra.utils.instantiate(cfg.encoder.encoder).to(device)
    if cfg.distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    linear_layer = hydra.utils.instantiate(cfg.encoder.linear_layer).to(device)
    if cfg.distributed:
        linear_layer = DDP(linear_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(encoder.parameters()) + list(linear_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=optim_params)

    learner = TemporalSSLLearner(
        optimizer = optimizer,
        repr_loss_fn = cfg.learner.repr_loss_fn,
        joint_diff_loss_fn = cfg.learner.joint_diff_loss_fn,
        encoder = encoder,
        linear_layer = linear_layer,
        joint_diff_scale_factor = cfg.learner.joint_diff_scale_factor,
        total_loss_type = cfg.learner.total_loss_type
    )
    learner.to(device)

    return learner

def init_bc(cfg, device, rank):
    image_encoder = hydra.utils.instantiate(cfg.encoder.image_encoder).to(device)
    if cfg.distributed:
        image_encoder = DDP(image_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    tactile_encoder = hydra.utils.instantiate(cfg.encoder.tactile_encoder).to(device)
    if cfg.distributed:
        tactile_encoder = DDP(tactile_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    last_layer = hydra.utils.instantiate(cfg.encoder.last_layer).to(device)
    if cfg.distributed:
        last_layer = DDP(last_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    learner = ImageTactileBC(
        image_encoder = image_encoder, 
        tactile_encoder = tactile_encoder,
        last_layer = last_layer,
        optimizer = optimizer,
        loss_fn = cfg.learner.loss_fn,
        representation_type = cfg.learner.representation_type,
        freeze_encoders = cfg.learner.freeze_encoders
    )
    learner.to(device) 
    
    return learner