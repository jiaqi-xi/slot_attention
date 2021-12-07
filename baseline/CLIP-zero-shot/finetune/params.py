import attr


@attr.s(auto_attribs=True)
class FinetuneParams:
    # architecture of CLIP pre-trained model
    clip_arch: str = 'ViT-B/16'
    freeze_text_encoder: bool = True

    # data
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    prompt: str = 'a photo of {color} {shape}'
    separater: str = ', '

    # training settings
    batch_size: int = 180
    num_workers: int = 6
    lr: float = 5e-5
    epochs: int = 100
