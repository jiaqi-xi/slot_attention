import json
import os
import cv2
import numpy as np
from PIL import Image
from typing import Callable
from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import clip


class CLEVRVisionLanguageCLIPDataset(Dataset):
    """Dataset that loads one random frame from CLEVR video.
    Also build one sentence (language) describing the video.
    """

    def __init__(self,
                 data_root: str,
                 clip_transforms: Callable,
                 split: str = "train",
                 clip_len: int = 34,
                 prompt: str = 'a {color} {shape}',
                 separater: str = ', '):
        super().__init__()
        self.data_root = data_root
        self.data_path = os.path.join(data_root, "images")
        self.split = split
        assert os.path.exists(
            self.data_root), f"Path {self.data_root} does not exist"
        assert self.split in ["train", "val"]
        assert os.path.exists(
            self.data_path), f"Path {self.data_path} does not exist"

        self.clip_transforms = clip_transforms

        with open(os.path.join('./data/', f'{split}_annos.json'), 'r') as f:
            self.anno_paths = json.load(f)
        self.anno_paths.sort()
        self.files, self.annos = self.get_files()

        self.num_videos = len(self.files)
        self.clip_len = clip_len
        self.prompt = prompt
        self.base_num = clip_len
        self.separater = separater

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""

        img = self._get_frame(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        text = clip.tokenize([text])[0]  # tokenize

        return dict(img=img, text=text)

    def _get_frame(self, index: int):
        """Get one random frame from the video."""
        img_idx, frame_idx = self._get_idx(index)
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, img = cap.read()
        assert success, f'read video {image_path} frame {frame_idx} failed!'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cap.release()
        # return the CLIP pre-processed image
        return self.clip_transforms(Image.fromarray(img))

    def _generate_text(self, index: int):
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        texts = [
            self.prompt.format(color=color, shape=shape)
            for color, shape in zip(colors, shapes)
        ]
        texts = self.separater.join(texts)
        return texts

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        img_paths, all_annos = [], []
        for anno_name in self.anno_paths:
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            if True:
                image_path = os.path.join(self.data_path,
                                          f"{anno['image_filename']}.avi")
                assert os.path.exists(
                    image_path), f"{image_path} does not exist"
                img_paths.append(image_path)
                all_annos.append(anno)
        return img_paths, all_annos

    def _get_idx(self, index):
        video_idx = index // self.base_num
        frame_idx = index % self.base_num
        return video_idx, frame_idx

    def __len__(self):
        return self.num_videos * self.base_num


class CLEVRVisionLanguageCLIPDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clip_transforms: Callable,
        num_workers: int,
        prompt: str = 'a {color} {shape}',
        separater: str = ', ',
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clip_transforms = clip_transforms
        self.num_workers = num_workers
        self.prompt = prompt
        self.separater = separater

        self.train_dataset = CLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            clip_transforms=self.clip_transforms,
            split='train',
            prompt=self.prompt,
            separater=self.separater,
        )
        self.val_dataset = CLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            clip_transforms=self.clip_transforms,
            split='val',
            prompt=self.prompt,
            separater=self.separater,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class DDPCLEVRVisionLanguageCLIPDataModule(CLEVRVisionLanguageCLIPDataModule):

    def __init__(self,
                 data_root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 clip_transforms: Callable,
                 num_workers: int,
                 prompt: str = 'a {color} {shape}',
                 separater: str = ', '):
        super().__init__(
            data_root,
            train_batch_size,
            val_batch_size,
            clip_transforms,
            num_workers,
            prompt=prompt,
            separater=separater)

    def train_dataloader(self):
        sampler = DistributedSampler(
            self.train_dataset, shuffle=True, drop_last=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )
