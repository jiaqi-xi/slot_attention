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
    dec_pos_enc: bool = True
    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1.0

    # architecture of CLIP pre-trained model
    use_clip_vision: bool = False
    clip_arch: str = 'ViT-B/32'
    enc_resolution: Tuple[int, int] = resolution  # image size
    clip_vision_channel: int = 64
    clip_text_channel: int = 512
    enc_pos_enc: bool = True

    # Text2Slot model
    use_text2slot: bool = True
    text2slot_arch: str = 'Transformer'
    # for MLP
    text2slot_hidden_sizes: Tuple[int] = (256, )
    predict_slot_dist: bool = False  # directly predict for each slot
    # for Transformer
    text2slot_hidden: int = 64
    text2slot_nhead: int = 1
    text2slot_num_transformers: int = 2
    text2slot_dim_feedforward: int = 256
    text2slot_dropout: float = 0.1
    text2slot_activation: str = 'relu'
    text2slot_text_pe: bool = True
    text2slot_padding_mask: bool = True
    text2slot_mlp_layers: int = 2

    # data
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    # Normalization for natural img or original slot attention one
    simple_normalize: bool = True  # since we not using ViT
    # whether load different text for different video period
    fine_grained: bool = True
    # whether text is complete action or just object names
    object_only: bool = False
    # separater to connect different words
    separater: str = ', '
    overfit: int = -1  # overfit to `overfit` data samples

    # training settings
    gpus: int = 1
    lr: float = 0.0004
    batch_size: int = 64
    val_batch_size: int = 64
    max_epochs: int = 8
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 4
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
