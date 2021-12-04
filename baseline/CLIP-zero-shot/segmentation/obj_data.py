import json
import os
import cv2
import numpy as np
from PIL import Image
from typing import Callable, Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import clip


def simple_rescale(x):
    return x * 2. - 1.


def build_data_transforms(params):
    _, clip_transforms = clip.load(params.clip_arch)
    if not params.use_clip_vision:
        from torchvision.transforms import Compose, Resize, ToTensor, \
            Normalize, Lambda
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        normalize = Lambda(
            simple_rescale) if params.simple_normalize else Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711))
        clip_transforms = Compose([
            Resize(params.resolution, interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            normalize,
        ])
    return clip_transforms


class CLEVRVisionLanguageCLIPDataset(Dataset):
    """Dataset that loads one random frame from CLEVR video.
    Also build one sentence (language) describing the video.
    """

    def __init__(
            self,
            data_root: str,
            max_num_images: Optional[int],
            clip_transforms: Callable,
            max_n_objects: int = 10,
            split: str = "train",
            clip_len: int = 34,
            is_video: bool = False,
            # whether generate separate texts for different time period
            # if False, just concat all three actions as one sentence
            fine_grained: bool = True,
            object_only: bool = False,  # only use "[color] [shape]" as text
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

        with open(os.path.join('./data/', f'{split}_annos.json'), 'r') as f:
            self.anno_paths = json.load(f)
        self.anno_paths.sort()
        self.files, self.annos = self.get_files()

        self.num_videos = len(self.files)
        self.clip_len = clip_len
        if self.split == 'train':
            self.base_num = clip_len
        else:
            self.base_num = 1
        self.is_video = is_video

        if object_only:
            # if object_only, we simply return all color-shape pairs for
            # every timestamp, so we shouldn't concat them
            assert fine_grained
        self.fine_grained = fine_grained
        self.object_only = object_only

        # pattern for text generation
        self.pattern0_1 = 'lift up the {color} {shape}'
        self.pattern0_2 = 'put down the {color} {shape}'
        self.pattern1 = 'put the {color1} {shape1} {pos} the {color2} {shape2}'
        self.pattern2 = 'put the {color1} {shape1} {pos} the {color2} {shape2}'
        self.POS = {
            0: 'on top of',
            1: 'in front of',
            2: 'to the right of',
            3: 'to the left of',
            4: 'behind',
        }
        self.separater = separater

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            video = self._get_video(index)  # clip pre-processed video frames
            raw_text = [
                self._generate_text(index) for _ in range(self.clip_len)
            ]  # raw
            text = clip.tokenize(raw_text)  # tokenize to [N, L]
            assert text.shape[0] == video.shape[0]
            if not self.fine_grained:  # raw_text are all the same
                raw_text = raw_text[0]
            else:  # need to concat to get the entire sentence
                raw_text = self.separater.join(
                    [raw_text[0], raw_text[4], raw_text[8], raw_text[21]])
            return dict(video=video, text=text, raw_text=raw_text)

        img, ori_img = self._get_frame(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        text = clip.tokenize([text])[0]  # tokenize

        return dict(ori_img=ori_img, img=img, text=text)

    def _get_frame(self, index: int):
        """Get one random frame from the video."""
        img_idx, frame_idx = self._get_idx(index)
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, img = cap.read()
        assert success, f'read video {image_path} frame {frame_idx} failed!'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_img = img.astype(np.int64)
        cap.release()
        # return the CLIP pre-processed image
        return self.clip_transforms(Image.fromarray(img)), ori_img

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
        return torch.stack(
            [self.clip_transforms(Image.fromarray(img)) for img in img_list],
            dim=0)

    def _generate_text(self, index: int):
        """Generate a sentence describing the video according to the annotation.

        Format of the scene json file
            - 'relationships': original CLEVR relation reasoning labels
            - 'image_index': int, the image to load, f'CLEVR_new_{idx:0=6d}.avi'
            - 'directions': euler angle representation of each direction
                e.g. 'behind': [-0.754490315914154, 0.6563112735748291, 0.0]
                    means in 3D space follows this direction
            - 'image_filename': str, filename of the video, f`{image_filename}.avi`
            - 'split': str, should always be 'new' here
            - 'actions': a list of length 3, an action of [a, b, c] means:
                0. lift up to show the first object and then put it down
                1. move the second object to the `a` direction of the first obj
                2. move the third object to the `c` direction of the `b`th obj
                    (Note that here we index from 1 not 0!)
                number-direction dict: {
                    0: 'up',
                    1: 'front',
                    2: 'right',
                    3: 'left',
                    4: 'behind'
                }
                Also for each video that has 34 frames, [0, 7] is showing the
                    first object, [8, 20] is doing the first action, [21, 33]
                    is doing the second action.
            - 'objects': a list of (single object) dict, the order of object
                is what we index them in the `actions` part. Keys:
                - 'pixel_coords': a list of 3 float, (x, y, depth) in image
                - '3d_coords': a list of 3 float, 3D coordinates in real-world
                - 'material': str, 'rubber', 'metal', etc.
                - 'shape': str, 'cylinder', 'cube', etc.
                - 'size': str, 'small', etc.
                - 'color': str, 'cyan', 'yellow', brown', etc.
                - 'rotation': float, rotation angle (degree) along Z-up axis
        """
        img_idx, frame_idx = self._get_idx(index)
        anno = self.annos[img_idx]
        object_colors = [obj['color'] for obj in anno['objects']]
        object_shapes = [obj['shape'] for obj in anno['objects']]
        actions = anno['actions']
        if self.object_only:
            '''
            text0_1 = text0_2 = f'{object_colors[0]} {object_shapes[0]}'
            text1 = f'{object_colors[1]} {object_shapes[1]}, ' \
                    f'{object_colors[0]} {object_shapes[0]}'
            text2 = f'{object_colors[2]} {object_shapes[2]}, ' \
                    f'{object_colors[actions[1] - 1]} ' \
                    f'{object_shapes[actions[1] - 1]}'
            '''
            # TODO: the comma here?
            # TODO: the order here may serve as contrastive learning signal
            # 'red cube, green cylinder, blue ball'
            text0_1 = text0_2 = text1 = text2 = self.separater.join([
                f'{color} {shape}'
                for color, shape in zip(object_colors, object_shapes)
            ])
        else:
            text0_1 = self.pattern0_1.format(
                color=object_colors[0], shape=object_shapes[0])
            text0_2 = self.pattern0_2.format(
                color=object_colors[0], shape=object_shapes[0])
            text1 = self.pattern1.format(
                color1=object_colors[1],
                shape1=object_shapes[1],
                pos=self.POS[actions[0]],
                color2=object_colors[0],
                shape2=object_shapes[0])
            text2 = self.pattern2.format(
                color1=object_colors[2],
                shape1=object_shapes[2],
                pos=self.POS[actions[2]],
                color2=object_colors[actions[1] - 1],
                shape2=object_shapes[actions[1] - 1])
        if not self.fine_grained:
            # just concat them with ', '
            text = self.separater.join([text0_1, text0_2, text1, text2])
            return text
        if 0 <= frame_idx <= 3:
            text = text0_1
        elif 4 <= frame_idx <= 7:
            text = text0_2
        elif 8 <= frame_idx <= 20:
            text = text1
        elif 21 <= frame_idx <= 33:
            text = text2
        else:
            raise NotImplementedError('Undefined frame index')
        return text

    def get_files(self) -> List[str]:
        """Load the image (video) path and loaded annotations (lists)."""
        img_paths, all_annos = [], []
        for i, anno_name in enumerate(self.anno_paths):
            if self.max_num_images is not None and \
                    len(img_paths) > self.max_num_images:
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
            num_objects = len(anno['objects'])
            # TODO: here we don't care about object num
            # if num_objects <= self.max_n_objects:
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
        if self.split != 'train' and not self.is_video:
            # random sample a frame_idx
            frame_idx = np.random.choice(self.clip_len)
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
            max_n_objects: int = 6,
            num_train_images: Optional[int] = None,
            num_val_images: Optional[int] = None,
            fine_grained: bool = True,
            object_only: bool = False,
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
        self.object_only = object_only
        self.fine_grained = fine_grained
        self.separater = separater
        self.overfit = overfit

        train_split = 'val' if self.overfit > 0 else 'train'
        self.train_dataset = CLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            split=train_split,
            max_n_objects=self.max_n_objects,
            fine_grained=self.fine_grained,
            object_only=self.object_only,
            separater=self.separater,
            overfit=self.overfit,
            repeat=(self.overfit > 0),
        )
        self.val_dataset = CLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clip_transforms=self.clip_transforms,
            split='val',
            max_n_objects=self.max_n_objects,
            fine_grained=self.fine_grained,
            object_only=self.object_only,
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
        is_video: bool = False,
        shuffle_obj: bool = False,
        pad_text: str = 'background',
    ):
        # TODO: we assume `self.max_n_objects` == 6 here!
        super().__init__(data_root, max_num_images, clip_transforms,
                         max_n_objects, split, clip_len, is_video, True, True)
        assert pad_text  # shouldn't be ''
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

        img, ori_img = self._get_frame(index)  # clip pre-processed img tensor
        text, obj_mask = self._generate_text(index)  # raw text
        tokens = self._tokenize_text(text)  # tokenize

        return dict(
            ori_img=ori_img,
            img=img,
            tokens=tokens,
            obj_mask=obj_mask,
            data_idx=index)

    def _tokenize_text(self, texts: Tuple[str]):
        """Tokenize texts and pad to `self.max_n_objects`"""
        assert len(texts) == self.text_num
        tokens = clip.tokenize(texts)  # [N + 1, C]
        return tokens

    def _generate_text(self, index: int, padding: bool = True):
        """Generate text descriptions of each object in the scene."""
        img_idx = self._get_idx(index)[0]
        anno = self.annos[img_idx]
        colors = [obj['color'] for obj in anno['objects']]
        shapes = [obj['shape'] for obj in anno['objects']]
        texts = [
            'a {} {}'.format(color, shape)
            for color, shape in zip(colors, shapes)
        ]
        # pad with some special texts, e.g. 'background'
        if padding:
            obj_mask = np.zeros(self.text_num, dtype=np.bool)
            obj_mask[:len(texts) + 1] = True  # we also need a background class
            texts = texts + [self.pad_text] * (self.text_num - len(texts))
            return texts, obj_mask
        texts.append(self.pad_text)
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
        pad_text: str = 'background',
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
