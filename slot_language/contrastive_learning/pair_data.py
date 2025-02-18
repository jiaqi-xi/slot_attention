import sys

sys.path.append('../')

import numpy as np

from data import CLEVRVisionLanguageCLIPDataset, CLEVRVisionLanguageCLIPDataModule


class PairCLEVRVisionLanguageCLIPDataset(CLEVRVisionLanguageCLIPDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: the text should be the same?
        assert self.object_only or not self.fine_grained
        # no use this here for code simplicity
        self.base_num = self.clip_len
        self.val_divide_num = 1

    def __getitem__(self, index: int):
        """Load one video and get only *TWO* frames from it"""
        if self.is_video:
            return super().__getitem__(index)

        video_idx, frame_idx = self._get_idx(index)
        paired_frame_idx = (np.random.choice(self.base_num - 1) +
                            (frame_idx + 1)) % self.base_num
        paired_index = video_idx * self.base_num + paired_frame_idx
        assert paired_frame_idx != frame_idx

        data1 = super().__getitem__(index)
        data2 = super().__getitem__(paired_index)
        assert (data1['text'] == data2['text']).all()

        data1.update(img2=data2['img'], text2=data2['text'])
        data1['video_idx'] = video_idx
        return data1


class PairCLEVRVisionLanguageCLIPDataModule(CLEVRVisionLanguageCLIPDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        train_split = 'val' if self.overfit > 0 else 'train'
        self.train_dataset = PairCLEVRVisionLanguageCLIPDataset(
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
        self.val_dataset = PairCLEVRVisionLanguageCLIPDataset(
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
