from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    # model configs
    # TODO: let's start with small img size!
    resolution: Tuple[int, int] = (64, 64)  # since we not using ViT
    num_slots: int = 7  # at most 6 obj per image/video
    # dim of slots embedding
    slot_size: int = 128
    num_iterations: int = 2
    # whether treat bg slot separately
    use_bg_sep_slot: bool = False
    # MLP hidden size in Slot Attention
    slot_mlp_size: int = 256  # FFN after cross attention
    # UNet as encoder
    use_unet: bool = False
    # Conv encoder-decoder
    out_features: int = 64
    dec_resolution: Tuple[int, int] = (8, 8)
    kernel_size: int = 5
    enc_channels: Tuple[int, ...] = (3, 64, 64, 64, 64)
    dec_channels: Tuple[int, ...] = (128, 64, 64, 64, 64)

    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1e-3

    # setting about sem-pos separate model
    use_sempos_sep: bool = True
    enc_pos_size: int = 64  # number of dims for positional information
    dec_pos_size: int = None  # if is int, then use cat instead of add

    # transformation equivariance loss
    flip_img: bool = True
    shuffle_obj: bool = False
    equivariance_loss_w: float = 1.0

    # contrastive loss on slot embedding
    use_contrastive_loss: bool = True
    contrastive_mlp: Tuple[int] = ()
    contrastive_T: float = 0.07
    contrastive_normalize: bool = True
    contrastive_stop_grad: bool = False
    contrastive_loss_w: float = 0.1

    # text reconstruction loss on slot embedding
    use_text_recon_loss: bool = False
    text_recon_mlp: Tuple[int] = (64, )
    text_recon_normalize: bool = False
    text_recon_loss_w: float = 0.01

    # feature loss, grouped regions should have similar feature vectors
    use_feature_loss: bool = False
    feature_loss_w: bool = 0.1

    # architecture of CLIP pre-trained model
    use_clip_vision: bool = False
    clip_arch: str = 'ViT-B/32'
    enc_resolution: Tuple[int, int] = resolution  # image size
    clip_vision_channel: int = 64
    clip_text_channel: int = 512

    # Text2Slot model
    use_text2slot: bool = True
    text2slot_arch: str = 'MLP'  # or 'Transformer' or 'DETR'
    # for MLP
    text2slot_hidden_sizes: Tuple[int] = (512, )
    normalize_slots: bool = True

    # data
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CATER/max2action/"
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video_4obj"
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video"
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    pad_text: str = 'background'
    prompt: str = 'a {color} {shape}'
    # Normalization for natural img or original slot attention one
    simple_normalize: bool = True  # since we not using ViT
    center_crop: Tuple[int] = None  # (192, 192)
    random_crop: Tuple[int] = None  # (192, 192)

    # training settings
    gpus: int = 1
    batch_size: int = 64
    val_batch_size: int = 64
    max_epochs: int = 16
    num_sanity_val_steps: int = 1
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 6
    n_samples: int = 5

    # optimization settings
    cosine_decay: bool = True
    lr: float = 0.0002
    warmup_steps_pct: float = 0.025
    decay_steps_pct: float = 0.2
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.2
