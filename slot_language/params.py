from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    # model configs
    resolution: Tuple[int, int] = (224, 224)
    num_slots: int = 7  # at most 6 obj per image/video
    # dim of slots embedding
    slot_size: int = 64
    num_iterations: int = 3
    # MLP hidden size in Slot Attention
    slot_mlp_size: int = 128
    dec_resolution: Tuple[int, int] = (7, 7)
    dec_kernel_size: int = 3
    dec_channels: Tuple[int, ...] = (64, 64, 64, 64, 64)
    # use self-entropy loss to masks
    use_entropy_loss: bool = False

    # architecture of CLIP pre-trained model
    clip_arch: str = 'ViT-B/32'
    clip_vision_channel: int = 768
    clip_text_channel: int = 512
    clip_global_feats: bool = False
    enc_pos_enc: bool = False

    # Text2Slot model
    text2slot_hidden_sizes: Tuple[int] = (256, )
    predict_slot_dist: bool = True

    # data
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    # whether load different text for different video period
    fine_grained: bool = True

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
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 4
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
