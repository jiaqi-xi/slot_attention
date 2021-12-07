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

    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clip_transforms: Callable,
        max_n_objects: int = 6,
        split: str = "train",
        clip_len: int = 34,
        prompt: str = 'a {color} {shape}',
        is_video: bool = False,
        shuffle_obj: bool = False,
        pad_text: str = 'background',
    ):
        # TODO: we assume `self.max_n_objects` == 6 here!
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, is_video, True, True)
        assert pad_text  # shouldn't be ''
        self.prompt = prompt
        self.shuffle_obj = shuffle_obj
        self.pad_text = pad_text
        self.text_num = 1 + self.max_n_objects

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            video = self._get_video(index)  # clip pre-processed video frames
            # TODO: since in CLEVR, text is the same through entire video
            # TODO: so I can simply repeat and stack them
            raw_text = [self._generate_text(index)] * self.clip_len  # raw
            tokens = [self._tokenize_text(text) for text in raw_text]
            return dict(
                video=video,  # [clip_len, C, H, W]
                text=torch.stack(tokens, dim=0),  # [clip_len, N + 1, C]
                raw_text=', '.join(raw_text[0]))  # one sentence

        img = self._get_frame(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        tokens = self._tokenize_text(text)  # tokenize

        return dict(img=img, text=tokens)

    def _tokenize_text(self, texts: Tuple[str]):
        """Tokenize texts and pad to `self.max_n_objects`"""
        assert len(texts) == self.text_num
        tokens = clip.tokenize(texts)  # [N + 1, C]
        return tokens

    def _generate_text(self, index: int):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        texts = [
            self.prompt.format(color=color, shape=shape)
            for color, shape in zip(colors, shapes)
        ]
        # pad with some special texts, e.g. 'background'
        texts = texts + [self.pad_text] * (self.text_num - len(texts))
        # shuffle the order of objects
        if self.split == 'train' and self.shuffle_obj:
            np.random.shuffle(texts)
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
        prompt: str = 'a {color} {shape}',
        shuffle_obj: bool = False,
        pad_text: str = 'background',
    ):
        super().__init__(data_root, train_batch_size, val_batch_size,
                         clip_transforms, num_workers, max_n_objects)

        self.prompt = prompt
        self.shuffle_obj = shuffle_obj
        self.pad_text = pad_text
        self.train_dataset = ObjCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='train',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
        )
        self.val_dataset = ObjCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='val',
            prompt=self.prompt,
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
            prompt: str = 'a {color} {shape}',
            is_video: bool = False,
            shuffle_obj: bool = False,
            pad_text: str = 'background',
            sample_clip_num: int = 2,  # loaded clips per video
    ):
        # TODO: we assume `self.max_n_objects` == 6 here!
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, prompt, is_video,
                         shuffle_obj, pad_text)
        self.sample_clip_num = sample_clip_num
        if self.split == 'train':
            self.base_num = self.clip_len - (self.sample_clip_num - 1)

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            data = super().__getitem__(index)
            data['text'] = data['text'][:1]
            return data

        clip = self._get_clip(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        tokens = self._tokenize_text(text)  # tokenize

        return dict(img=clip, text=tokens)

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

    def _get_idx(self, index):
        video_idx = index // self.base_num
        frame_idx = index % self.base_num
        if self.split != 'train' and not self.is_video:
            # random sample a frame_idx
            frame_idx = np.random.choice(self.clip_len - self.sample_clip_num)
        return video_idx, frame_idx


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
        prompt: str = 'a {color} {shape}',
        shuffle_obj: bool = False,
        pad_text: str = 'background',
        sample_clip_num: int = 2,
    ):
        super().__init__(data_root, train_batch_size, val_batch_size,
                         clip_transforms, num_workers, max_n_objects, prompt,
                         shuffle_obj, pad_text)

        self.sample_clip_num = sample_clip_num
        self.train_dataset = ObjRecurCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='train',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
            sample_clip_num=self.sample_clip_num,
        )
        self.val_dataset = ObjRecurCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='val',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
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
                 prompt: str = 'a {color} {shape}',
                 is_video: bool = False,
                 shuffle_obj: bool = False,
                 pad_text: str = 'background',
                 flip_img: bool = False):
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, prompt, is_video,
                         shuffle_obj, pad_text)
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
            tokens = self._tokenize_text(text)
            return dict(
                img=img,
                flipped_img=flipped_img,
                is_flipped=self.flip_img,
                text=tokens)

        # load text description and potentially do text shuffling
        text, shuffled_text, shuffled_idx = self._generate_text(index)
        tokens = self._tokenize_text(text)
        shuffled_tokens = self._tokenize_text(shuffled_text)
        return dict(
            img=img,
            flipped_img=flipped_img,
            is_flipped=self.flip_img,
            text=tokens,
            shuffled_text=shuffled_tokens,
            shuffled_idx=shuffled_idx,
            is_shuffled=self.shuffle_obj)

    def _generate_text(self, index: int):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        texts = [
            self.prompt.format(color=color, shape=shape)
            for color, shape in zip(colors, shapes)
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
        prompt: str = 'a {color} {shape}',
        shuffle_obj: bool = False,
        pad_text: str = 'background',
        flip_img: bool = False,
    ):
        super().__init__(data_root, train_batch_size, val_batch_size,
                         clip_transforms, num_workers, max_n_objects, prompt,
                         shuffle_obj, pad_text)

        self.flip_img = flip_img
        self.train_dataset = ObjAugCLEVRVisionLanguageCLIPDataset(
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
        self.val_dataset = ObjAugCLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            max_n_objects=self.max_n_objects,
            split='val',
            prompt=self.prompt,
            shuffle_obj=self.shuffle_obj,
            pad_text=self.pad_text,
            flip_img=self.flip_img,
        )
