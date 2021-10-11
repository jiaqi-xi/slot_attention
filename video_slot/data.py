import json
import os
import cv2
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import compact


class CLEVRVideoFrameDataset(Dataset):
    """Dataset that loads one random frame from CLEVR video"""

    def __init__(
            self,
            data_root: str,
            max_num_images: Optional[int],
            clevr_transforms: Callable,
            max_n_objects: int = 10,
            split: str = "train",
            clip_len: int = 34,  # TODO: assume each video has same length!
            is_video: bool = False,  # if True, return the entire video
            sample_clip_num: int = 2,  # loaded clips per video
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "images")
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(
            self.data_root), f"Path {self.data_root} does not exist"
        assert self.split in ["train", "val"]
        assert os.path.exists(
            self.data_path), f"Path {self.data_path} does not exist"

        with open(os.path.join('./data/', f'{split}_annos.json'), 'r') as f:
            self.anno_paths = json.load(f)
        self.files = self.get_files()

        self.num_videos = len(self.files)
        self.clip_len = clip_len
        self.is_video = is_video
        self.sample_clip_num = sample_clip_num

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            return self._get_video(index)

        # since we take subseq of video frames
        img_idx = index // (self.clip_len - (self.sample_clip_num - 1))
        frame_idx = index % (self.clip_len - (self.sample_clip_num - 1))
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        imgs = []
        for _ in range(self.sample_clip_num):
            success, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert success, f'read video {image_path} frame {frame_idx} fail!'
            imgs.append(img)
        cap.release()
        # return shape [sample_clip_num, 3, H, W]
        return torch.stack([self.clevr_transforms(img) for img in imgs], dim=0)

    def __len__(self):
        return len(self.files) * (self.clip_len - (self.sample_clip_num - 1))

    def _get_video(self, index: int):
        # assume input is video index!
        img_idx = index
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        success = True
        img_list = []
        while success:
            success, img = cap.read()
            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_list.append(img)
        cap.release()
        return torch.stack([self.clevr_transforms(img) for img in img_list],
                           dim=0)

    def get_files(self) -> List[str]:
        paths = []
        for anno_name in self.anno_paths:
            if self.max_num_images is not None and \
                    len(paths) > self.max_num_images:
                break
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            num_objects = len(anno['objects'])
            if num_objects <= self.max_n_objects:
                image_path = os.path.join(self.data_path,
                                          f"{anno['image_filename']}.avi")
                assert os.path.exists(
                    image_path), f"{image_path} does not exist"
                paths.append(image_path)
        return sorted(compact(paths))


class CLEVRVideoFrameDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clevr_transforms: Callable,
        max_n_objects: int,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
        sample_clip_num: int = 2,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = CLEVRVideoFrameDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
            sample_clip_num=sample_clip_num,
        )
        self.val_dataset = CLEVRVideoFrameDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            split="val",
            max_n_objects=self.max_n_objects,
            sample_clip_num=sample_clip_num,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CLEVRTransforms(object):

    def __init__(self, resolution: Tuple[int, int]):
        '''
        crop = ((29, 221), (64, 256))
        # TODO: whether to add center crop here?
        transforms.Lambda(
            lambda X: X[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]),
        '''
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            transforms.Lambda(
                lambda X: 2 * X - 1.0),  # rescale between -1 and 1
            transforms.Resize(resolution),
        ])

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
