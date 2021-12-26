import json
import os
import cv2
import numpy as np
from typing import Callable
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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

        self.files = self.get_files()

        self.num_videos = len(self.files)
        self.clip_len = clip_len
        self.is_video = is_video
        self.sample_clip_num = sample_clip_num
        if self.split == 'train':
            self.base_num = self.clip_len - (self.sample_clip_num - 1)
        else:
            self.base_num = 1  # in val, one clip per video

    def _rand_another(self, index):
        another_index = np.random.choice(len(self))
        return self.__getitem__(another_index)

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            return self._get_video(index)

        # since we take subseq of video frames
        img_idx, frame_idx = self._get_idx(index)
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        imgs = []
        for _ in range(self.sample_clip_num):
            success, img = cap.read()
            if not success:
                cap.release()
                return self._rand_another(index)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        cap.release()
        # return shape [sample_clip_num, 3, H, W]
        return torch.stack([self.clevr_transforms(img) for img in imgs], dim=0)

    def __len__(self):
        return self.num_videos * self.base_num

    def _get_idx(self, index):
        img_idx = index // self.base_num
        frame_idx = index % self.base_num
        if self.split != 'train' and not self.is_video:
            # random sample a frame_idx
            frame_idx = np.random.choice(self.clip_len - self.sample_clip_num)
        return img_idx, frame_idx

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
        if not len(img_list):  # empty video
            return self._rand_another(index)
        return torch.stack([self.clevr_transforms(img) for img in img_list],
                           dim=0)

    def get_files(self) -> List[str]:
        with open(os.path.join('data/', f'{self.split}_annos.json'), 'r') as f:
            self.anno_paths = json.load(f)
        paths = []
        for anno_name in self.anno_paths:
            if self.max_num_images is not None and \
                    len(paths) > self.max_num_images:
                break
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            num_objects = len(anno['objects'])
            # TODO: here we don't care about object num
            # if num_objects <= self.max_n_objects:
            if True:
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
        self.sample_clip_num = sample_clip_num

        self._build_dataset()

    def _build_dataset(self):
        self.train_dataset = CLEVRVideoFrameDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
            sample_clip_num=self.sample_clip_num,
        )
        self.val_dataset = CLEVRVideoFrameDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            split="val",
            max_n_objects=self.max_n_objects,
            sample_clip_num=self.sample_clip_num,
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


class CATERVideoFrameDataset(CLEVRVideoFrameDataset):
    """Dataset that loads one random frame from CATER video"""

    def __init__(
            self,
            data_root: str,
            max_num_images: Optional[int],
            clevr_transforms: Callable,
            max_n_objects: int = 10,
            split: str = "train",
            clip_len: int = 301,  # TODO: assume each video has same length!
            is_video: bool = False,  # if True, return the entire video
            sample_clip_num: int = 6,  # loaded clips per video
    ):
        super().__init__(data_root, max_num_images, clevr_transforms,
                         max_n_objects, split, clip_len, is_video,
                         sample_clip_num)

    def get_files(self) -> List[str]:
        with open(
                os.path.join('data/', f'cater_{self.split}_annos.json'),
                'r') as f:
            self.anno_paths = json.load(f)
        paths = []
        for anno_name in self.anno_paths:
            if self.max_num_images is not None and \
                    len(paths) > self.max_num_images:
                break
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            num_objects = len(anno['objects'])
            # TODO: here we don't care about object num
            # if num_objects <= self.max_n_objects:
            if True:
                img_name = anno['image_filename'].replace(
                    'CLEVR_new', 'CATER_new')
                image_path = os.path.join(self.data_path, img_name)
                assert os.path.exists(
                    image_path), f"{image_path} does not exist"
                paths.append(image_path)
        return sorted(compact(paths))


class CATERVideoFrameDataModule(CLEVRVideoFrameDataModule):

    def _build_dataset(self):
        self.train_dataset = CATERVideoFrameDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            max_n_objects=self.max_n_objects,
            split="train",
            sample_clip_num=self.sample_clip_num,
        )
        self.val_dataset = CATERVideoFrameDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            max_n_objects=self.max_n_objects,
            split="val",
            sample_clip_num=self.sample_clip_num,
        )
