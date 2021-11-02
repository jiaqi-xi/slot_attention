import json
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class YouCook2FrameDataset(Dataset):
    """Dataset that loads one from from YouCook2 Video."""

    def __init__(self,
                 data_root: str,
                 youcook2_transforms: Callable,
                 split: str = 'train',
                 is_video: bool = False,
                 overfit: int = -1):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.data_root = data_root
        self.youcook2_transforms = youcook2_transforms
        self.split = split
        self.is_video = is_video
        # TODO: if overfit >= 1, then only load these data and repeat them
        # TODO: if overfit <= 0, then no overfitting
        assert isinstance(overfit, int)
        if overfit >= 1:
            print(f'Training data overfitting to {overfit} examples')
        self.overfit = overfit

        path_name = f'{self.split}ing' if self.split in ['train', 'test'] \
            else 'validation'
        self.data_path = os.path.join(self.data_root, 'raw_videos', path_name)
        # a csv file with ['vid_id', 'duration', 'total_frame'] columns
        self.video_stats = pd.read_csv(
            os.path.join(self.data_root, 'splits',
                         f'{self.split}_duration_totalframe.csv'))
        # each lines are ['405/Ysh60eirChU', '405/jpQBWsR3HHs', ...]
        self.data_files = np.loadtxt(
            os.path.join(self.data_root, 'splits', f'{self.split}_list.txt'),
            dtype='<U15')
        # a dict, keys are 'vid_id', dict['vid_id'] is still a dict with keys
        # ['duration', 'subset', 'recipe_type', 'annotations', 'video_url']
        # dict['vid_id']['annotations'] is a list of dict describing actions
        # this dict has keys ['segment', 'id', 'sentence']:
        #   - 'segment': ['starting_second', 'ending_second'] of this action
        #   - 'id': indicating this is the `id`-th action in the video
        #   - 'sentence': the natural language description of this action
        anno_name = 'trainval' if self.split in ['train', 'val'] \
            else 'test_segments_only'
        with open(
                os.path.join(self.data_root,
                             f'youcookii_annotations_{anno_name}.json'),
                'r') as f:
            self.annos = json.load(f)['database']

        self.paths, self.stats, self.annos = self._get_files()
        self.num_videos = len(self.paths)
        # get mapping from frame_idx to video_idx
        self.frame_idx = [0]
        for stat in self.stats:
            self.frame_idx.append(self.frame_idx[-1] + stat['total_frame'])
        self.total_frames = self.frame_idx[-1]

    def _frame_idx2video_idx(self, frame_idx: int):
        for i in range(len(self.frame_idx) - 1):
            if self.frame_idx[i] <= frame_idx < self.frame_idx[i + 1]:
                return i
        return None

    def _rand_another(self):
        """Randomly get another data if current video is unavailable."""
        if self.is_video:
            new_idx = np.random.choice(self.num_videos)
        else:
            new_idx = np.random.choice(self.total_frames)
        return self.__getitem__(new_idx)

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            return self._get_video(index)

        video_idx = self._frame_idx2video_idx(index)
        frame_idx = index - self.frame_idx[video_idx]
        video_path = self.paths[video_idx]
        vid_id = self.stats[video_idx]['vid_id']
        # TODO: if frames have been extracted, read from img
        img_path = os.path.join(
            os.path.dirname(video_path).replace('raw_videos', 'raw_imgs'),
            vid_id, f'{vid_id}_{frame_idx:06d}.jpg')
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            img = img.convert("RGB")
        else:
            # if img not existed, read one frame from video
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, img = cap.read()
            cap.release()
            if not success:
                return self._rand_another()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.youcook2_transforms(img)

    def __len__(self):
        return self.total_frames

    def _get_video(self, index: int):
        # assume input is video index!
        video_idx = index
        video_path = self.paths[video_idx]
        img_list = []
        vid_id = self.stats[video_idx]['vid_id']
        # TODO: if frames have been extracted, read from img
        img_folder = os.path.join(
            os.path.dirname(video_path).replace('raw_videos', 'raw_imgs'),
            vid_id)
        if os.path.exists(img_folder):
            all_imgs = [
                fn for fn in os.listdir(img_folder) if fn.endswith('.jpg')
            ]
            all_imgs.sort()
            for filename in all_imgs:
                img_path = os.path.join(img_folder, filename)
                img = Image.open(img_path)
                img = img.convert("RGB")
                img_list.append(img)
        else:
            cap = cv2.VideoCapture(video_path)
            success = True
            while success:
                success, img = cap.read()
                if success:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_list.append(img)
            cap.release()
        if len(img_list) == 0:
            return self._rand_another()
        return torch.stack([self.youcook2_transforms(img) for img in img_list],
                           dim=0)

    def _get_files(self):
        paths, stats, annos = [], [], []
        for i, filename in enumerate(self.data_files):
            if self.overfit >= 1:
                filename = self.data_files[i % self.overfit]
            # 'xxx/youcook/raw_videos/training/405/Ysh60eirChU'
            video_name = os.path.join(self.data_path, filename)
            possible_path = [
                f'{video_name}{suffix}'
                for suffix in ['', '.mp4', '.mkv', '.webm']
            ]
            video_path = None
            for path in possible_path:
                if os.path.exists(path):
                    video_path = path
                    break
            if video_path is None:  # video not exist
                continue
            paths.append(video_path)
            # get video statistics
            vid_id = filename.split('/')[-1]  # 'Ysh60eirChU'
            stat = self.video_stats[self.video_stats['vid_id'] == vid_id]
            duration = float(stat['duration'])
            total_frame = int(stat['total_frame'])
            fps = total_frame / duration
            stat = dict(
                vid_id=vid_id,
                duration=duration,
                total_frame=total_frame,
                fps=fps)
            stats.append(stat)
            # get annotations
            annos.append(self.annos[vid_id]['annotations'])  # a list of dicts
        return paths, stats, annos


class YouCook2FrameDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_root: str,
            train_batch_size: int,
            val_batch_size: int,
            youcook2_transforms: Callable,
            num_workers: int,
            overfit: int = -1,  # overfit to one training example
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.youcook2_transforms = youcook2_transforms
        self.num_workers = num_workers
        self.overfit = overfit

        train_split = 'val' if self.overfit > 0 else 'train'
        self.train_dataset = YouCook2FrameDataset(
            data_root=self.data_root,
            youcook2_transforms=self.youcook2_transforms,
            split=train_split,
            overfit=self.overfit,
        )
        self.val_dataset = YouCook2FrameDataset(
            data_root=self.data_root,
            youcook2_transforms=self.youcook2_transforms,
            split='val',
            overfit=self.overfit,
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
