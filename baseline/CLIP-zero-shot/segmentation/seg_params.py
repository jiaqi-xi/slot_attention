import attr


@attr.s(auto_attribs=True)
class SegParams:
    # architecture of CLIP pre-trained model
    clip_arch: str = 'ViT-B/16'

    # data
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video_4obj"
    # data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/CLEVR_viewpoint_video"
    data_root: str = "/scratch/ssd004/scratch/ziyiwu/data/clevr_video/train/"
    max_n_objects: int = 6

    # training settings
    batch_size: int = 64
    num_workers: int = 6
    num_test: int = 8
    prompt: str = 'a photo of {color} {shape}'
    pad_text: str = 'empty'

    # for segmentation mask visualization
    PALETTE = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
               [255, 0, 255], [100, 100, 255], [200, 200, 100],
               [170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100],
               [200, 200, 200], [50, 50, 50]]
