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
    # MLP hidden size in Slot Attention
    slot_mlp_size: int = 128  # FFN after cross attention
    dec_resolution: Tuple[int,
                          int] = (resolution[0] // 16, resolution[1] // 16)
    dec_kernel_size: int = 5
    dec_channels: Tuple[int, ...] = tuple(64 for _ in range(4))
    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1.0
    equivariance_loss_w: float = 1.0
    # whether treat bg slot separately
    use_bg_sep_slot: bool = False

    # whether use pos slot attention model
    use_slot_pos_emb: bool = False
    num_pos_slot: int = 4
    share_pos_slot: bool = False

    # whether use unet slot attention model
    use_unet_slot_model: bool = False
    kernel_size: int = dec_kernel_size
    enc_channels: Tuple[int, ...] = (64, 64, 64, 64, 64)
    # dec_channels: Tuple[int, ...] = (64, 64, 64, 64, 64)

    # architecture of CLIP pre-trained model
    use_clip_vision: bool = False
    clip_arch: str = 'ViT-B/32'
    enc_resolution: Tuple[int, int] = resolution  # image size
    clip_vision_channel: int = 64
    clip_text_channel: int = 512
    enc_pos_enc: bool = True

    # Text2Slot model
    use_text2slot: bool = True
    text2slot_arch: str = 'MLP'  # or 'Transformer' or 'DETR'
    # for MLP
    text2slot_hidden_sizes: Tuple[int] = (512, )

    # data
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    shuffle_obj: bool = False
    flip_img: bool = True
    # Normalization for natural img or original slot attention one
    simple_normalize: bool = True  # since we not using ViT

    # training settings
    gpus: int = 4
    lr: float = 0.001
    batch_size: int = 64 * 4
    val_batch_size: int = 64 * 4
    max_epochs: int = 16
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 6
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
