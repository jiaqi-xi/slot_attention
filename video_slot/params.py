from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.0004
    batch_size: int = 64
    val_batch_size: int = 64
    resolution: Tuple[int, int] = (128, 128)
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
    # whether use relu in SlotModel
    use_relu: bool = True
    # MLP hidden size in Slot Attention
    slot_mlp_size = 128
    # eval epoch interval
    eval_interval = 1.0
    # mixed precision training
    fp16 = False
    # sample clips per video as input
    sample_clip_num = 2
    # predict mask or image
    pred_mask = True
    # whether stop the gradient of GT future
    stop_future_grad = False
