import os
import copy
import json
import numpy as np
from typing import Callable, List
from typing import Optional

from obj_data import ObjCLEVRVisionLanguageCLIPDataset, ObjCLEVRVisionLanguageCLIPDataModule, \
    ObjAugCLEVRVisionLanguageCLIPDataset, ObjAugCLEVRVisionLanguageCLIPDataModule


class ObjCATERVisionLanguageCLIPDataset(ObjCLEVRVisionLanguageCLIPDataset):

    def __init__(self,
                 data_root: str,
                 max_num_images: Optional[int],
                 clip_transforms: Callable,
                 tokenizer: str = 'clip',
                 max_n_objects: int = 10,
                 split: str = "train",
                 clip_len: int = 301,
                 prompt: str = 'a {color} {shape}',
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 pad_text: str = 'background'):
        self.cater_subset = 'cater_cameramotion' if \
            'cameramotion' in data_root else 'cater'
        super().__init__(
            data_root,
            max_num_images,
            clip_transforms,
            tokenizer=tokenizer,
            max_n_objects=max_n_objects,
            split=split,
            clip_len=clip_len,
            prompt=prompt,
            is_video=is_video,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text)

    def _generate_text(self, index: int):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        sizes = [obj['size'] for obj in anno['objects']]
        # e.g. 'a large red cone'
        texts = [
            self.prompt.format(size=size, color=color, shape=shape)
            for size, color, shape in zip(sizes, colors, shapes)
        ]
        # pad with some special texts, e.g. 'background'
        texts = texts + [self.pad_text] * (self.text_num - len(texts))
        # shuffle the order of objects
        if self.split == 'train' and self.shuffle_obj:
            np.random.shuffle(texts)
        return texts

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        with open(
                os.path.join('./data/',
                             f'{self.cater_subset}_{self.split}_annos.json'),
                'r') as f:
            self.anno_paths = json.load(f)
        self.anno_paths.sort()
        img_paths, all_annos = [], []
        for i, anno_name in enumerate(self.anno_paths):
            if self.max_num_images is not None and \
                    len(img_paths) > self.max_num_images:
                break
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            img_name = anno['image_filename'].replace('CLEVR_new', 'CATER_new')
            image_path = os.path.join(self.data_path, img_name)
            assert os.path.exists(image_path), f"{image_path} does not exist"
            img_paths.append(image_path)
            all_annos.append(anno)
        return img_paths, all_annos


class ObjCATERVisionLanguageCLIPDataModule(ObjCLEVRVisionLanguageCLIPDataModule
                                           ):

    def __init__(self,
                 data_root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 clip_transforms: Callable,
                 num_workers: int,
                 tokenizer: str = 'clip',
                 max_n_objects: int = 10,
                 prompt: str = 'a {color} {shape}',
                 shuffle_obj: bool = False,
                 pad_text: str = 'background'):
        super().__init__(
            data_root,
            train_batch_size,
            val_batch_size,
            clip_transforms,
            num_workers,
            tokenizer=tokenizer,
            max_n_objects=max_n_objects,
            prompt=prompt,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text)

    def _build_dataset(self):
        self.train_dataset = ObjCATERVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            tokenizer=self.tokenizer,
            max_n_objects=self.max_n_objects,
            split='train',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
        )
        self.val_dataset = ObjCATERVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.val_clip_transforms,
            tokenizer=self.tokenizer,
            max_n_objects=self.max_n_objects,
            split='val',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
        )


class ObjAugCATERVisionLanguageCLIPDataset(ObjAugCLEVRVisionLanguageCLIPDataset
                                           ):

    def __init__(self,
                 data_root: str,
                 max_num_images: Optional[int],
                 clip_transforms: Callable,
                 tokenizer: str = 'clip',
                 max_n_objects: int = 10,
                 split: str = "train",
                 clip_len: int = 301,
                 prompt: str = 'a {color} {shape}',
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 pad_text: str = 'background',
                 flip_img: bool = False):
        self.cater_subset = 'cater_cameramotion' if \
            'cameramotion' in data_root else 'cater'
        super().__init__(
            data_root,
            max_num_images,
            clip_transforms,
            tokenizer=tokenizer,
            max_n_objects=max_n_objects,
            split=split,
            clip_len=clip_len,
            prompt=prompt,
            is_video=is_video,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text,
            flip_img=flip_img)

    def _generate_text(self, index: int):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        sizes = [obj['size'] for obj in anno['objects']]
        texts = [
            self.prompt.format(size=size, color=color, shape=shape)
            for size, color, shape in zip(sizes, colors, shapes)
        ]
        # pad with some special texts, e.g. 'background'
        texts = texts + [self.pad_text] * (self.text_num - len(texts))
        # shuffle the order of objects
        if self.split == 'train':
            idx = np.arange(len(texts))
            if self.shuffle_obj:
                np.random.shuffle(idx)
                shuffled_texts = [texts[i] for i in idx]
            else:
                shuffled_texts = copy.deepcopy(texts)
            return texts, shuffled_texts, idx
        return texts

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        with open(
                os.path.join('./data/',
                             f'{self.cater_subset}_{self.split}_annos.json'),
                'r') as f:
            self.anno_paths = json.load(f)
        self.anno_paths.sort()
        img_paths, all_annos = [], []
        for i, anno_name in enumerate(self.anno_paths):
            if self.max_num_images is not None and \
                    len(img_paths) > self.max_num_images:
                break
            anno_path = os.path.join(self.data_root, 'scenes', anno_name)
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            img_name = anno['image_filename'].replace('CLEVR_new', 'CATER_new')
            image_path = os.path.join(self.data_path, img_name)
            assert os.path.exists(image_path), f"{image_path} does not exist"
            img_paths.append(image_path)
            all_annos.append(anno)
        return img_paths, all_annos


class ObjAugCATERVisionLanguageCLIPDataModule(
        ObjAugCLEVRVisionLanguageCLIPDataModule):

    def __init__(self,
                 data_root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 clip_transforms: Callable,
                 num_workers: int,
                 tokenizer: str = 'clip',
                 max_n_objects: int = 10,
                 prompt: str = 'a {color} {shape}',
                 shuffle_obj: bool = False,
                 pad_text: str = 'background',
                 flip_img: bool = False):
        super().__init__(
            data_root,
            train_batch_size,
            val_batch_size,
            clip_transforms,
            num_workers,
            tokenizer=tokenizer,
            max_n_objects=max_n_objects,
            prompt=prompt,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text,
            flip_img=flip_img)

    def _build_dataset(self):
        self.train_dataset = ObjAugCATERVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            tokenizer=self.tokenizer,
            max_n_objects=self.max_n_objects,
            split='train',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
            flip_img=self.flip_img,
        )
        self.val_dataset = ObjAugCATERVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.val_clip_transforms,
            tokenizer=self.tokenizer,
            max_n_objects=self.max_n_objects,
            split='val',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
            flip_img=self.flip_img,
        )
