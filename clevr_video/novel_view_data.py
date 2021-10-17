import os
from PIL import Image
from typing import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class CLEVRNovelViewImageDataset(Dataset):
    """Dataset that loads image from one view from CLEVR novel view image."""

    def __init__(self,
                 data_root: str,
                 clevr_transforms: Callable,
                 split: str = '00'):
        assert split in ['00', '01']
        self.split = split
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.data_path = os.path.join(data_root, "images")
        assert os.path.exists(
            self.data_root), f"Path {self.data_root} does not exist"
        assert os.path.exists(
            self.data_path), f"Path {self.data_path} does not exist"

        self.pairs = self.get_pairs()

    def __getitem__(self, index: int):
        """Load one view image"""
        pair_name = self.pairs[index]
        img = os.path.join(self.data_path, f'{pair_name}{self.split}.png')
        img = Image.open(img)
        img = img.convert("RGB")
        return self.clevr_transforms(img)

    def __len__(self):
        return len(self.pairs)

    def get_pairs(self):
        all_files = os.listdir(self.data_path)
        all_files = list(set([file[:-6] for file in all_files]))
        # file is like 'CLEVR_new_000007'
        # so a pair is f'{file}00.png' and f'{file}01.png'
        return all_files


class CLEVRNovelViewImagePairDataset(CLEVRNovelViewImageDataset):
    """Dataset that loads paired images from CLEVR novel view image."""

    def __init__(self, data_root: str, clevr_transforms: Callable):
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.data_path = os.path.join(data_root, "images")
        assert os.path.exists(
            self.data_root), f"Path {self.data_root} does not exist"
        assert os.path.exists(
            self.data_path), f"Path {self.data_path} does not exist"

        self.pairs = self.get_pairs()

    def __getitem__(self, index: int):
        """Load two views paired image"""
        pair_name = self.pairs[index]
        img1 = os.path.join(self.data_path, f'{pair_name}00.png')
        img2 = os.path.join(self.data_path, f'{pair_name}01.png')
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")
        return torch.stack(
            [self.clevr_transforms(img) for img in [img1, img2]], dim=0)


class CLEVRNovelViewImageDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clevr_transforms: Callable,
        num_workers: int,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.num_workers = num_workers

        self.train_dataset = CLEVRNovelViewImageDataset(
            data_root=self.data_root,
            clevr_transforms=self.clevr_transforms,
            split='00',
        )
        self.val_dataset = CLEVRNovelViewImagePairDataset(
            data_root=self.data_root,
            clevr_transforms=self.clevr_transforms,
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
