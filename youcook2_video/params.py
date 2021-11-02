from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    # training settings
    lr: float = 0.0004
    gpus: int = 1
    max_epochs: int = 8
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2

    # data settings
    data_root: str = "/scratch/ssd004/datasets/youcook/"
    num_workers: int = 8
    batch_size: int = 64
    val_batch_size: int = 64
    overfit: int = -1

    # model settings
    resolution: Tuple[int, int] = (128, 128)
    num_slots: int = 7  # 5 change to 7 according to official code
    num_iterations: int = 3  # MLP hidden size in Slot Attention
    slot_mlp_size: int = 128
    # use self-entropy loss to masks
    use_entropy_loss: bool = False
    entropy_loss_w: float = 1.0
    # whether set the slot parameters as learnable (to be updated by BP)
    # TODO: should be True in official code!!!
    # TODO: but this codebase set it as False and I've done lots of exp using
    # TODO: it so far... So I set False as the default value
    learnable_slot: bool = False
