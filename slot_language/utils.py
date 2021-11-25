import cv2
import numpy as np
from PIL import Image
from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from pytorch_lightning import Callback

import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def conv_transpose_out_shape(in_size,
                             stride,
                             padding,
                             kernel_size,
                             out_padding,
                             dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (
        kernel_size - 1) + out_padding + 1


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]],
                 expected: Tuple[int, ...],
                 message: str = ""):
    assert actual == expected, \
        f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images, masks = pl_module.sample_images()
                trainer.logger.experiment.log(
                    {"images": [wandb.Image(images),
                                wandb.Image(masks)]},
                    commit=False)


class PosSlotImageLogCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images, masks, all_masks = pl_module.sample_images()
                trainer.logger.experiment.log(
                    {
                        "images": [
                            wandb.Image(images),
                            wandb.Image(masks),
                            wandb.Image(all_masks)
                        ]
                    },
                    commit=False)


class TwoStreamImageLogCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images, masks, coarse_masks = pl_module.sample_images()
                trainer.logger.experiment.log(
                    {
                        "images": [
                            wandb.Image(images),
                            wandb.Image(masks),
                            wandb.Image(coarse_masks)
                        ]
                    },
                    commit=False)


class VideoLogCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                video, text = pl_module.sample_video()
                video = (video * 255.).numpy().astype(np.uint8)
                trainer.logger.experiment.log(
                    {"video": [wandb.Video(video, fps=2, caption=text)]},
                    commit=False)


def simple_rescale(x: Tensor) -> Tensor:
    return x * 2. - 1.


def to_rgb_from_tensor(x: Tensor,
                       mean=(0.48145466, 0.4578275, 0.40821073),
                       std=(0.26862954, 0.26130258, 0.27577711)):
    """Reverse the Normalize operation in torchvision."""
    if -1. <= x.min().item() <= x.max().item() <= 1.:
        # this is from simple_rescale
        return (x * 0.5 + 0.5).clamp(0, 1)
    assert len(x.shape) == 5, f'x shape {x.shape} is not [B, slots, 3, H, W]'
    for i in range(3):
        x[:, :, i].mul_(std[i]).add_(mean[i])
    return x.clamp(0, 1)


def save_video(video, save_path, fps=30, codec='mp4v'):
    """video: np.ndarray of shape [M, H, W, 3]"""
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    assert video.dtype in [np.float16, np.float32, np.float64]  # [0., 1.]
    assert len(video.shape) == 4, 'unsupported save video shape'
    assert video.shape[-1] == 3  # colored video
    # TODO: cv2 has different color channel order GBR
    video = video[..., [2, 1, 0]]
    H, W = video.shape[-3:-1]
    # video = np.ascontiguousarray(video)
    assert save_path.split('.')[-1] == 'mp4'  # save as mp4 file
    # clip max value
    # if video.max() > 1.:
    #     print('Warning! Video max value > 1.0')
    # video = np.clip(video, a_min=0., a_max=1.)
    # make video uint8 array for save
    video = np.round(video * 255.).astype(np.uint8)
    # opencv has opposite dimension definition as numpy
    size = [W, H]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*codec), fps, size)
    for i in range(video.shape[0]):
        out.write(video[i])
    out.release()
