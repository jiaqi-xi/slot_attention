import sys
import cv2
import copy
import numpy as np
from PIL import Image
from typing import Callable, Tuple
from typing import Optional
import torch
import torchvision.transforms.functional as TF

import clip

sys.path.append('../')

from data import CLEVRVisionLanguageCLIPDataset, CLEVRVisionLanguageCLIPDataModule


class ObjCLEVRVisionLanguageCLIPDataset(CLEVRVisionLanguageCLIPDataset):
    """Dataset that loads one random frame from CLEVR video.
    One text ('color-shape' of an object) directly for one slot!
    """

    def __init__(self,
                 data_root: str,
                 max_num_images: Optional[int],
                 clip_transforms: Callable,
                 max_n_objects: int = 6,
                 split: str = "train",
                 clip_len: int = 34,
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 pad_text: str = ''):
        # TODO: we assume `self.max_n_objects` == 6 here!
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, is_video, True, True)
        self.shuffle_obj = shuffle_obj
        self.pad_text = pad_text

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            video = self._get_video(index)  # clip pre-processed video frames
            raw_text = [
                self._generate_text(index * self.base_num + idx)
                for idx in range(self.base_num)
            ]  # raw
            token = [self._pad_text_tokens(text) for text in raw_text]
            return dict(
                video=video,
                text=torch.stack([t[0] for t in token], dim=0),
                padding=torch.stack([t[1] for t in token], dim=0),
                raw_text=', '.join(raw_text[0]))

        img = self._get_frame(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        token, padding = self._pad_text_tokens(text)  # tokenize

        return dict(img=img, text=token, padding=padding)

    def _pad_text_tokens(self, texts: Tuple[str]):
        """Tokenize texts and pad to `self.max_n_objects`"""
        tokens = clip.tokenize(texts)  # [n, C]
        # TODO: we're using `+1` to count for the background slot
        num_pad = 1 + self.max_n_objects - tokens.shape[0]
        pad_tokens = torch.zeros(num_pad, tokens.shape[1], dtype=tokens.dtype)
        padding = torch.cat(
            [torch.ones(tokens.shape[0]),
             torch.zeros(num_pad)], dim=0).long()
        return torch.cat([tokens, pad_tokens], dim=0), padding

    def _generate_text(self, index: int):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        texts = [
            'a {} {}'.format(color, shape)
            for color, shape in zip(colors, shapes)
        ]
        # shuffle the order of objects
        if self.split == 'train' and self.shuffle_obj:
            np.random.shuffle(texts)
        if self.pad_text:  # pad with some special texts e.g. 'no object'
            texts = texts + [
                self.pad_text,
            ] * (1 + self.max_n_objects - len(texts))
        return texts


class ObjCLEVRVisionLanguageCLIPDataModule(CLEVRVisionLanguageCLIPDataModule):

    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clip_transforms: Callable,
        num_workers: int,
        max_n_objects: int = 6,
        shuffle_obj: bool = False,
        pad_text: str = '',
    ):
        super().__init__(data_root, train_batch_size, val_batch_size,
                         clip_transforms, num_workers, max_n_objects)

        self.shuffle_obj = shuffle_obj
        self.pad_text = pad_text
        self.train_dataset = ObjCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='train',
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
        )
        self.val_dataset = ObjCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='val',
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
        )


class ObjRecurCLEVRVisionLanguageCLIPDataset(ObjCLEVRVisionLanguageCLIPDataset
                                             ):
    """Dataset that loads *consequent* frames (clips) from CLEVR video.
    One text ('color-shape' of an object) directly for one slot!
    """

    def __init__(
            self,
            data_root: str,
            max_num_images: Optional[int],
            clip_transforms: Callable,
            max_n_objects: int = 6,
            split: str = "train",
            clip_len: int = 34,
            is_video: bool = False,
            shuffle_obj: bool = False,
            sample_clip_num: int = 2,  # loaded clips per video
    ):
        # TODO: we assume `self.max_n_objects` == 6 here!
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, is_video, shuffle_obj)
        self.sample_clip_num = sample_clip_num
        if self.split == 'train':
            self.base_num = self.clip_len - (self.sample_clip_num - 1)
        else:
            self.base_num = self.clip_len
        self.val_divide_num = 2 * self.sample_clip_num if \
            self.split == 'val' else 1

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            data = super().__getitem__(index)
            data['text'] = data['text'][:1]
            data['padding'] = data['padding'][:1]
            return data

        clip = self._get_clip(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        token, padding = self._pad_text_tokens(text)  # tokenize

        return dict(img=clip, text=token, padding=padding)

    def _get_clip(self, index: int):
        """Get one random frame from the video."""
        img_idx, frame_idx = self._get_idx(index)
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        imgs = []
        for _ in range(self.sample_clip_num):
            success, img = cap.read()
            assert success, f'read video {image_path} frame {frame_idx} fail!'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        cap.release()
        # return the CLIP pre-processed image
        # of shape [`sample_clip_num`, 3, H, W]
        return torch.stack(
            [self.clip_transforms(Image.fromarray(img)) for img in imgs],
            dim=0)


class ObjRecurCLEVRVisionLanguageCLIPDataModule(
        ObjCLEVRVisionLanguageCLIPDataModule):

    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clip_transforms: Callable,
        num_workers: int,
        max_n_objects: int = 6,
        shuffle_obj: bool = False,
        sample_clip_num: int = 2,
    ):
        super().__init__(data_root, train_batch_size, val_batch_size,
                         clip_transforms, num_workers, max_n_objects,
                         shuffle_obj)

        self.sample_clip_num = sample_clip_num
        self.train_dataset = ObjRecurCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='train',
            shuffle_obj=self.shuffle_obj,
            sample_clip_num=self.sample_clip_num,
        )
        self.val_dataset = ObjRecurCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='val',
            shuffle_obj=self.shuffle_obj,
            sample_clip_num=self.sample_clip_num,
        )


class ObjAugCLEVRVisionLanguageCLIPDataset(ObjCLEVRVisionLanguageCLIPDataset):
    """Dataset that loads one random frame from CLEVR video.
    One text ('color-shape' of an object) directly for one slot!
    Apply random augmentation to get another view of the frame.
    """

    def __init__(self,
                 data_root: str,
                 max_num_images: Optional[int],
                 clip_transforms: Callable,
                 max_n_objects: int = 6,
                 split: str = "train",
                 clip_len: int = 34,
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 flip_img: bool = False):
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, is_video, shuffle_obj)
        self.flip_img = flip_img

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            return super().__getitem__(index)

        # load one frame and potentially do horizontal flip
        img = self._get_frame(index)  # clip pre-processed img tensor
        if self.flip_img:
            flipped_img = TF.hflip(img)
        else:
            flipped_img = img.detach().clone()
        if self.split != 'train':
            text = self._generate_text(index)
            token, padding = self._pad_text_tokens(text)
            return dict(
                img=img,
                flipped_img=flipped_img,
                is_flipped=self.flip_img,
                text=token,
                padding=padding)

        # load text description and potentially do text shuffling
        text, shuffled_texts, shuffled_idx = self._generate_text(index)
        token, padding = self._pad_text_tokens(text)
        shuffled_token, shuffled_padding, shuffled_idx = self._pad_text_tokens(
            shuffled_texts, shuffled_idx)
        assert (padding == shuffled_padding).all()
        return dict(
            img=img,
            flipped_img=flipped_img,
            is_flipped=self.flip_img,
            text=token,
            padding=padding,
            shuffled_text=shuffled_token,
            shuffled_idx=shuffled_idx,
            is_shuffled=self.shuffle_obj)

    def _pad_text_tokens(self, texts: Tuple[str], text_idx: np.ndarray = None):
        """Tokenize texts and pad to `self.max_n_objects`"""
        tokens = clip.tokenize(texts)  # [n, C]
        # TODO: we're using `+1` to count for the background slot
        num_pad = 1 + self.max_n_objects - tokens.shape[0]
        pad_tokens = torch.zeros(num_pad, tokens.shape[1], dtype=tokens.dtype)
        padded_tokens = torch.cat([tokens, pad_tokens], dim=0)
        padding = torch.cat(
            [torch.ones(tokens.shape[0]),
             torch.zeros(num_pad)], dim=0).long()
        if text_idx is not None:  # [n]
            padded_text_idx = -np.ones(padding.shape[0]).astype(np.int32)
            padded_text_idx[:text_idx.shape[0]] = text_idx
            return padded_tokens, padding, padded_text_idx
        return padded_tokens, padding

    def _generate_text(self, index: int):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        texts = [
            'a {} {}'.format(color, shape)
            for color, shape in zip(colors, shapes)
        ]
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


class ObjAugCLEVRVisionLanguageCLIPDataModule(
        ObjCLEVRVisionLanguageCLIPDataModule):

    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clip_transforms: Callable,
        num_workers: int,
        max_n_objects: int = 6,
        shuffle_obj: bool = False,
        flip_img: bool = False,
    ):
        super().__init__(data_root, train_batch_size, val_batch_size,
                         clip_transforms, num_workers, max_n_objects,
                         shuffle_obj)

        self.flip_img = flip_img
        self.train_dataset = ObjAugCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='train',
            shuffle_obj=self.shuffle_obj,
            flip_img=self.flip_img,
        )
        self.val_dataset = ObjAugCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='val',
            shuffle_obj=self.shuffle_obj,
            flip_img=self.flip_img,
        )
