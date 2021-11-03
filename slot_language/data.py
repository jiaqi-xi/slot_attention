import json
import os
import cv2
import clip
from PIL import Image
from typing import Callable
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CLEVRVisionLanguageCLIPDataset(Dataset):
    """Dataset that loads one random frame from CLEVR video
    Also build one sentence (language) describing the video.
    """

    def __init__(
            self,
            data_root: str,
            max_num_images: Optional[int],
            clip_transforms: Callable,
            max_n_objects: int = 10,
            split: str = "train",
            clip_len: int = 34,  # TODO: assume each video has same length!
            is_video: bool = False,  # if True, return the entire video
            # whether generate separate texts for different time period
            # if False, just concat all three actions as one sentence
        fine_grained: bool = True,
            object_only: bool = False,  # only use "[color] [shape]" as text
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
        self.is_video = is_video
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

    def __getitem__(self, index: int):
        """Load one video and get only one frame from it"""
        if self.is_video:
            video = self._get_video(index)  # clip pre-processed video frames
            raw_text = [
                self._generate_text(index * self.clip_len + idx)
                for idx in range(self.clip_len)
            ]  # raw
            text = clip.tokenize(raw_text)  # tokenize to [N, L]
            assert text.shape[0] == video.shape[0]
            if not self.fine_grained:
                raw_text = raw_text[0]
            else:
                raw_text = f'{raw_text[0]}, {raw_text[4]}, ' \
                           f'{raw_text[8]}, {raw_text[21]}'
            return dict(video=video, text=text, raw_text=raw_text)

        img = self._get_frame(index)  # clip pre-processed img tensor
        text = self._generate_text(index)  # raw text
        text = clip.tokenize([text])[0]  # tokenize

        return dict(img=img, text=text)

    def _get_frame(self, index: int):
        """Get one random frame from the video."""
        img_idx, frame_idx = index // self.clip_len, index % self.clip_len
        image_path = self.files[img_idx]
        cap = cv2.VideoCapture(image_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert success, f'read video {image_path} frame {frame_idx} failed!'
        cap.release()
        # return the CLIP pre-processed image
        return self.clip_transforms(Image.fromarray(img))

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
        img_idx, frame_idx = index // self.clip_len, index % self.clip_len
        anno = self.annos[img_idx]
        object_colors = [obj['color'] for obj in anno['objects']]
        object_shapes = [obj['shape'] for obj in anno['objects']]
        actions = anno['actions']
        if self.object_only:
            text0_2 = text0_1 = f'{object_colors[0]} {object_shapes[0]}'
            text1 = f'{object_colors[1]} {object_shapes[1]}, ' \
                    f'{object_colors[0]} {object_shapes[0]}'
            text2 = f'{object_colors[2]} {object_shapes[2]}, ' \
                    f'{object_colors[actions[1] - 1]} ' \
                    f'{object_shapes[actions[1] - 1]}'
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
            # just concat them with ','
            text = f'{text0_1}, {text0_2}, {text1}, {text2}'
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

    def __len__(self):
        return len(self.files) * self.clip_len


class CLEVRVisionLanguageCLIPDataModule(pl.LightningDataModule):

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
            fine_grained: bool = True,
            object_only: bool = False,
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
        self.fine_grained = fine_grained
        self.overfit = overfit

        train_split = 'val' if self.overfit > 0 else 'train'
        self.train_dataset = CLEVRVisionLanguageCLIPDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clip_transforms=self.clip_transforms,
            split=train_split,
            max_n_objects=self.max_n_objects,
            fine_grained=self.fine_grained,
            object_only=object_only,
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
            object_only=object_only,
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
