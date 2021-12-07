from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    # model configs
    resolution: Tuple[int, int] = (64, 64)
    kernel_size: int = 5
    out_features: int = 64
    enc_hiddens: Tuple[int, ...] = (3, 32, 32, 32, 32)
    dec_hiddens: Tuple[int, ...] = (128, 64, 64, 64, 64)
    decoder_resolution: Tuple[int, int] = (8, 8)
    use_unet: bool = False
    use_deconv: bool = True

    # slot attention module
    slot_size: int = 128
    num_slots: int = 7
    num_iterations: int = 2
    # MLP hidden size in Slot Attention
    slot_mlp_size: int = 256
    # whether set the slot parameters as learnable (to be updated by BP)
    learnable_slot = True

    # perform recurrent slot-attention
    recur_predictor: str = ''  # currently support ['', 'MLP']
    stop_recur_slot_grad: bool = False

    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1.0

    # data
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CATER/max2action/"
    # sample clips per video as input
    sample_clip_num: int = 6  # from the video slot-attn paper

    # training settings
    gpus: int = 4
    batch_size: int = 16 * 4
    val_batch_size: int = 16 * 4
    num_workers: int = 6
    max_epochs: int = 16
    num_sanity_val_steps: int = 1
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    n_samples: int = 5

    # optimization settings
    cosine_decay: bool = True
    lr: float = 2e-4
    warmup_steps_pct: float = 0.025
    decay_steps_pct: float = 0.2
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.05
