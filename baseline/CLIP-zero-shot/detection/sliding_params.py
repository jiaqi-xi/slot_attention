from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlidingParams:
    # model configs
    resolution: Tuple[int, int] = (256, 256)  # since we not using ViT
    num_slots: int = 7  # at most 6 obj per image/video

    # architecture of CLIP pre-trained model
    clip_arch: str = 'ViT-B/32'

    # data
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video"
    data_root: str = "/scratch/ssd004/scratch/jiaqixi/data/clevr_video/train/"
    shuffle_obj: bool = False
    # Normalization for natural img or original slot attention one
    simple_normalize: bool = False  # since we not using ViT
