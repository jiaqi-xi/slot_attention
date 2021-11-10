import json
import os
import cv2
import clip
from PIL import Image
from typing import Callable, Tuple
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CLEVRVisionLanguageViewpointDataset(Dataset):
    """Dataset that loads one random frame from CLEVR video.
    Also build one sentence (language) describing the video.
    """

    def __init__(self,
                 data_root: str,
                 max_num_images: Optional[int],
                 clip_transforms: Callable,
                 max_n_objects: int = 2,
                 split: str = "train",
                 clip_len: int = 11,
                 is_video: bool = False,
                 separater: str = ', ',
                 overfit: int = -1,
                 repeat: bool = False):
        super().__init__()
        self.data_root = data_root
        self.data_path = os.path.join(data_root, "images")
        self.split = split
        assert os.path.exists(
            self.data_root), f"Path {self.data_root} does not exist"
        assert self.split in ["train", "val"]
        assert os.path.exists(
            self.data_path), f"Path {self.data_path} does not exist"

        self.max_num_images = max_num_images
        self.clip_transforms = clip_transforms
        self.max_n_objects = max_n_objects

        # TODO: if overfit >= 1, then only load these data and repeat them
        # TODO: if overfit <= 0, then no overfitting
        assert isinstance(overfit, int)
        if overfit >= 1:
            print(f'Training data overfitting to {overfit} examples')
        self.overfit = overfit
        # TODO: whether to repeat data to match normal number
        # TODO: true for train set, False for test set
        if repeat:
            assert overfit >= 1
        self.repeat = repeat

        with open(
                os.path.join('./data/', f'viewpoint_{split}_annos.json'),
                'r') as f:
            self.anno_paths = json.load(f)
        self.anno_paths.sort()
        self.files, self.annos = self.get_files()

        self.num_videos = len(self.files)
        self.clip_len = clip_len
        self.base_num = self.clip_len  # for video/frame_idx calculation
        self.is_video = is_video

        # pattern for text generation
        # TODO: assume there are only 2 objects in each scene
        self.pattern = separater.join(
            ['{color1} {shape1}', '{color2} {shape2}'])

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            video = self._get_video(index)  # clip pre-processed video frames
            raw_text = [
                self._generate_text(index * self.base_num + idx)
                for idx in range(self.base_num)
            ]  # raw
            text = clip.tokenize(raw_text)  # tokenize to [N, L]
            assert text.shape[0] == video.shape[0]
            raw_text = raw_text[0]
            return dict(video=video, text=text, raw_text=raw_text)

        img = self._get_frame(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        text = clip.tokenize([text])[0]  # tokenize

        return dict(img=img, text=text)

    def _get_frame(self, index: int):
        """Get one random frame from the video."""
        img_idx, frame_idx = self._get_idx(index)
        image_folder = self.files[img_idx]
        image_path = os.path.join(image_folder, f'{frame_idx:02d}.png')
        # return the CLIP pre-processed image
        return self.clip_transforms(Image.open(image_path))

    def _get_video(self, index: int):
        # assume input is video index!
        img_idx = index
        image_folder = self.files[img_idx]
        img_list = [
            os.path.join(image_folder, f'{i:02d}.png')
            for i in range(self.clip_len)
        ]
        return torch.stack(
            [self.clip_transforms(Image.open(img)) for img in img_list], dim=0)

    def _generate_text(self, index: int):
        """Generate a sentence describing the video."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        text = self.pattern.format(
            color1=colors[0],
            shape1=shapes[0],
            color2=colors[1],
            shape2=shapes[1])
        return text

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        img_folders, all_annos = [], []
        for i, anno_name in enumerate(self.anno_paths):
            if self.max_num_images is not None and \
                    len(img_folders) > self.max_num_images:
                break
            if self.overfit >= 1:
                # no repeating for testing
                if i >= self.overfit and not self.repeat:
                    break
                # repeat for training
                anno_name = self.anno_paths[i % self.overfit]
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            # num_objects = len(anno['objects'])
            # TODO: here we don't care about object num
            # if num_objects <= self.max_n_objects:
            if True:
                image_folder = os.path.join(self.data_path,
                                            f"{anno['image_filename']}")
                assert os.path.exists(
                    image_folder), f"{image_folder} does not exist"
                img_folders.append(image_folder)
                all_annos.append(anno)
        return img_folders, all_annos

    def _get_idx(self, index):
        video_idx = index // self.base_num
        frame_idx = index % self.base_num
        return video_idx, frame_idx

    def __len__(self):
        return len(self.files) * self.base_num


class CLEVRVisionLanguageViewpointDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_root: str,
            train_batch_size: int,
            val_batch_size: int,
            clip_transforms: Callable,
            max_n_objects: int,
            num_workers: int,
            num_train_images: Optional[int] = None,
            num_val_images: Optional[int] = None,
            separater: str = ', ',
            overfit: int = -1,  # overfit to one training example
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clip_transforms = clip_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images
        self.separater = separater
        self.overfit = overfit

        train_split = 'val' if self.overfit > 0 else 'train'
        self.train_dataset = CLEVRVisionLanguageViewpointDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            split=train_split,
            max_n_objects=self.max_n_objects,
            separater=self.separater,
            overfit=self.overfit,
            repeat=(self.overfit > 0),
        )
        self.val_dataset = CLEVRVisionLanguageViewpointDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            split='val',
            max_n_objects=self.max_n_objects,
            separater=self.separater,
            overfit=self.overfit,
            repeat=False,
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
