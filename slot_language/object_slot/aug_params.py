from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    # model configs
    resolution: Tuple[int, int] = (128, 128)  # since we not using ViT
    num_slots: int = 7  # at most 6 obj per image/video
    # dim of slots embedding
    slot_size: int = 64
    num_iterations: int = 3
    # whether treat bg slot separately
    use_bg_sep_slot: bool = False
    # MLP hidden size in Slot Attention
    slot_mlp_size: int = 128  # FFN after cross attention
    # UNet as encoder
    use_unet: bool = False
    # Conv encoder-decoder
    out_features: int = 64
    dec_resolution: Tuple[int, int] = (8, 8)
    kernel_size: int = 5
    enc_channels: Tuple[int, ...] = (3, 64, 64, 64, 64)
    dec_channels: Tuple[int, ...] = (64, 64, 64, 64, 64)

    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1.0

    # setting about sem-pos separate model
    use_sempos_sep: bool = False
    enc_pos_size: int = 64  # number of dims for positional information
    dec_pos_size: int = None  # if is int, then use cat instead of add

    # transformation equivariance loss
    flip_img: bool = True
    equivariance_loss_w: float = 1.0

    # contrastive loss on slot embedding
    use_contrastive_loss: bool = False
    contrastive_T: float = 1.0

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
    normalize_slots: bool = False

    # data
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video_4obj"
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video"
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    shuffle_obj: bool = False
    pad_text: str = 'background'
    # Normalization for natural img or original slot attention one
    simple_normalize: bool = True  # since we not using ViT

    # training settings
    gpus: int = 4
    batch_size: int = 64 * 4
    val_batch_size: int = 64 * 4
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
    lr: float = 0.0008
    warmup_steps_pct: float = 0.025
    decay_steps_pct: float = 0.2
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.2
