import os
import json
from typing import Callable, List
from typing import Optional

from obj_data import ObjCLEVRVisionLanguageCLIPDataset, ObjCLEVRVisionLanguageCLIPDataModule, \
    ObjAugCLEVRVisionLanguageCLIPDataset, ObjAugCLEVRVisionLanguageCLIPDataModule


class ObjCATERVisionLanguageCLIPDataset(ObjCLEVRVisionLanguageCLIPDataset):

    def __init__(self,
                 data_root: str,
                 max_num_images: Optional[int],
                 clip_transforms: Callable,
                 max_n_objects: int = 10,
                 split: str = "train",
                 clip_len: int = 301,
                 prompt: str = 'a {color} {shape}',
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 pad_text: str = 'background'):
        super().__init__(
            data_root,
            max_num_images,
            clip_transforms,
            max_n_objects=max_n_objects,
            split=split,
            clip_len=clip_len,
            prompt=prompt,
            is_video=is_video,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text)

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        with open(
                os.path.join('./data/', f'cater_{self.split}_annos.json'),
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
            max_n_objects=max_n_objects,
            prompt=prompt,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text)

    def _build_dataset(self):
        self.train_dataset = ObjCATERVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
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
                 max_n_objects: int = 10,
                 split: str = "train",
                 clip_len: int = 301,
                 prompt: str = 'a {color} {shape}',
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 pad_text: str = 'background',
                 flip_img: bool = False):
        super().__init__(
            data_root,
            max_num_images,
            clip_transforms,
            max_n_objects=max_n_objects,
            split=split,
            clip_len=clip_len,
            prompt=prompt,
            is_video=is_video,
            shuffle_obj=shuffle_obj,
            pad_text=pad_text,
            flip_img=flip_img)

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        with open(
                os.path.join('./data/', f'cater_{self.split}_annos.json'),
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
            max_n_objects=self.max_n_objects,
            split='val',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
            flip_img=self.flip_img,
        )
