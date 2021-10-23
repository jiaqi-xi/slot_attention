from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.0004
    batch_size: int = 64
    val_batch_size: int = 64
    resolution: Tuple[int, int] = (128, 128)
    kernel_size: int = 5
    hidden_dims: Tuple[int, ...] = (64, 64, 64, 64)
    decoder_resolution: Tuple[int, int] = (8, 8)
    use_deconv: bool = True
    slot_size: int = 64
    num_slots: int = 7  # 5 change to 7 according to official code
    num_iterations: int = 3
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    gpus: int = 1
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
    # MLP hidden size in Slot Attention
    slot_mlp_size: int = 128
    # sample clips per video as input
    sample_clip_num: int = 2
    # predict mask or image
    pred_mask: bool = False
    # whether stop the gradient of GT future
    stop_future_grad: bool = False
    # whether and what to use as perceptual loss
    perceptual_loss: str = 'alex'
    # whether set the slot parameters as learnable (to be updated by BP)
    # TODO: should be True in official code!!!
    # TODO: but this codebase set it as False and I've done lots of exp using
    # TODO: it so far... So I set False as the default value
    learnable_slot = False
    # perform recurrent slot-attention
    recurrent_slot_attention: bool = False
    stop_recur_slot_grad: bool = False
    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1.0
