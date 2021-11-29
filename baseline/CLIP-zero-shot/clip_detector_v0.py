import os
import sys
import importlib
import argparse
import cv2
import numpy as np
from typing import Optional
from itertools import chain, combinations

import clip
import torch
from torchvision.transforms import Resize
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from ensemble_boxes import weighted_boxes_fusion

from obj_data import ObjCLEVRVisionLanguageCLIPDataset
from sliding_params import SlidingParams

sys.path.append('../../slot_language/')

from train import build_data_transforms


class AnchorImageDataset(Dataset):

    def __init__(self, image, coords, transforms):
        self.image = image.copy()
        self.coords = coords
        self.transforms = transforms

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return self.transforms(self.image.crop(coord))


def build_dataset(params):
    clip_transforms = build_data_transforms(params)
    clevr_dataset = ObjCLEVRVisionLanguageCLIPDataset(
        data_root=params.data_root,
        max_num_images=params.num_train_images,
        clip_transforms=clip_transforms,
        max_n_objects=params.max_n_objects,
        split='train',
        shuffle_obj=params.shuffle_obj,
    )
    return clevr_dataset


def get_img(index: int, dataset):
    img_idx, frame_idx = dataset._get_idx(index)
    image_path = dataset.files[img_idx]
    cap = cv2.VideoCapture(image_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, img = cap.read()
    assert success, f'read video {image_path} frame {frame_idx} failed!'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()
    return img


def get_anchor_coords(image_w, image_h, count_w, count_h):
    step_w = image_w // count_w + 1
    step_h = image_h // count_h + 1

    w_coords = np.arange(0, image_w + step_w, step_w)
    h_coords = np.arange(0, image_h + step_h, step_h)

    coords = []
    for x1, x2 in zip(w_coords[:-1], w_coords[1:]):
        x1, x2 = min(x1, image_w), min(x2, image_w)
        for y1, y2 in zip(h_coords[:-1], h_coords[1:]):
            y1, y2 = min(y1, image_h), min(y2, image_h)
            coords.append((x1, y1, x2, y2))

            anchor_w = (x2 - x1) / 2
            anchor_h = (y2 - y1) / 2
            anchor_xc = (x2 + x1) // 2
            anchor_yc = (y2 + y1) // 2

            for coef_x, coef_y in [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (1, 2), (2, 1),
                (2, 3), (3, 2),
                (2, 4), (4, 2),
                (3, 1), (1, 3),
                (5, 4), (4, 5),
                (4, 1), (1, 4),
                (5, 1), (1, 5),
                (5, 3), (3, 5),
                (6, 4), (4, 6),
                (5, 8), (8, 5),
                (10, 2), (2, 10),
                (10, 4), (4, 10),
                (10, 6), (6, 10),
                (10, 8), (8, 10),
            ]:
                anc_x1 = max(anchor_xc - (anchor_w * coef_x), 0)
                anc_x2 = min(anchor_xc + (anchor_w * coef_x), image_w)
                anc_y1 = max(anchor_yc - (anchor_h * coef_y), 0)
                anc_y2 = min(anchor_yc + (anchor_h * coef_y), image_h)
                coords.append((anc_x1, anc_y1, anc_x2, anc_y2))

    return coords


class CLIPDetectorV0:

    def __init__(self, model, transforms, device):
        """
        First version, required improving of quality and speed :)
        """
        self.device = device
        self.model = model
        self.transforms = transforms
        self.model.to(device)
        self.model.eval()
        # self.zero_text_embeddings = self.model.encode_text(
        #     clip.tokenize([template.format('') for template in IMAGENET_TEMPLATES]).to(self.device)
        # )

    def get_anchor_features(self, img, coords, bs=32, quite=False):
        anchor_dataset = AnchorImageDataset(img, coords, self.transforms)
        anchor_loader = DataLoader(
            anchor_dataset,
            batch_size=bs,
            sampler=SequentialSampler(anchor_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=2,
        )
        if not quite:
            anchor_loader = tqdm(anchor_loader)
        anchor_features = []
        for anchor_batch in anchor_loader:
            with torch.no_grad():
                anchor_features_ = self.model.encode_image(anchor_batch.to(self.device))
                anchor_features_ /= anchor_features_.norm(dim=-1, keepdim=True)
                anchor_features.append(anchor_features_)
        return torch.vstack(anchor_features)

    def draw(self, img, results, label='', colour=(0, 0, 255), width=1, font_colour=(0, 0, 0), font_scale=1,
             font_thickness=1,
             T=0.6):
        """
        :param img:
        :param results:
        :param label:
        :param colour:
        :param width:
        :param font_scale:
        :param font_thickness:
        :param T: transparency
        :return:
        """
        img = np.array(img)
        R, G, B = colour

        for score, (x1, y1, x2, y2) in zip(results['scores'], results['boxes']):
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colour, width)
            img = cv2.putText(img, f'{label}:{score:.3f}', (x1 + (x2 - x1) // 10, y1 + (y2 - y1) // 5), cv2.FONT_ITALIC,
                              font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
            img = cv2.putText(img, f'{label}:{score:.3f}', (x1 + (x2 - x1) // 10, y1 + (y2 - y1) // 5), cv2.FONT_ITALIC,
                              font_scale, font_colour, font_thickness, cv2.LINE_AA)
        return img

    def detect_by_text(
            self, texts, img, coords, anchor_features, *, tp_thr=0.0, fp_thr=-2.0, iou_thr=0.01, skip_box_thr=0.1, k=1
    ):
        """
        :param texts: list of text query
        :param img: PIL of raw image
        :param coords: list of anchor coords Pascal/VOC format [[x1,y1,x2,y2], ...]
        :param anchor_features: pt tensor with anchor features of image
        :param tp_thr: threshold of true positive query, uses for full size image
        :param fp_thr: threshold of false positive query, uses for full size image
        :param iou_thr: parameter for ensemble boxes using WBF, see https://github.com/ZFTurbo/Weighted-Boxes-Fusion
        :param skip_box_thr: parameter for ensemble boxes using WBF, see https://github.com/ZFTurbo/Weighted-Boxes-Fusion
        :param k: output top-k results
        :return: (img, result, thr)
        """
        zeroshot_weights = []
        with torch.no_grad():
            text_embeddings = []
            for text in texts:
                tokens = clip.tokenize([text]).to(self.device)
                text_embeddings.append(self.model.encode_text(tokens))

            text_embeddings = torch.stack(text_embeddings).mean(0)
            # text_embeddings -= self.zero_text_embeddings
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.mean(dim=0)
            text_embeddings /= text_embeddings.norm()

            zeroshot_weights.append(text_embeddings)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
            logits = (anchor_features @ zeroshot_weights).reshape(-1)
            probas, indexes = torch.sort(logits, descending=True)

            img_features = self.model.encode_image(self.transforms(img).unsqueeze(0).to(self.device)).squeeze(0)
            thr = (img_features @ zeroshot_weights)[0].item()

        w, h = img.size
        boxes_list = []
        scores_list = []
        labels_list = []
        probas = probas.cpu().numpy()
        probas = probas - np.min(probas)
        probas = probas / max(0.2, np.max(probas))
        # print(probas)
        print(f'thr = {thr}')
        thr_indexes = np.argpartition(probas, -k)[-k:]
        print(f'thr_indexes = {thr_indexes}')

        if thr > fp_thr:
            if thr_indexes.shape[0] != 0:
                for best_index, proba in zip(indexes[thr_indexes], probas[thr_indexes]):
                    x1, y1, x2, y2 = list(coords[best_index])
                    x1, y1, x2, y2 = max(x1 / w, 0.0), max(y1 / h, 0.0), min(x2 / w, 1.0), min(y2 / h, 1.0)
                    boxes_list.append([x1, y1, x2, y2])
                    scores_list.append(proba)
                    labels_list.append(1)
            else:
                if thr > tp_thr:
                    best_index, proba = indexes[0], probas[0]
                    x1, y1, x2, y2 = list(coords[best_index])
                    x1, y1, x2, y2 = max(x1 / w, 0.0), max(y1 / h, 0.0), min(x2 / w, 1.0), min(y2 / h, 1.0)
                    boxes_list.append([x1, y1, x2, y2])
                    scores_list.append(proba)
                    labels_list.append(1)

        boxes, scores, labels = weighted_boxes_fusion(
            [boxes_list], [scores_list], [labels_list],
            weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        result = {'boxes': [], 'scores': [], 'labels': []}
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            result['boxes'].append([x1, y1, x2, y2])
            result['scores'].append(float(score))
            result['labels'].append(int(label))
            # draw = ImageDraw.Draw(img)
            # draw.rectangle((x1, y1, x2, y2), width=2, outline=(0, 0, 255))
        # return img, result, thr
        print('result boxes = ', result['boxes'])
        return result


def main(params: Optional[SlidingParams] = None):
    if params is None:
        params = SlidingParams()

    # sample an image from the dataset
    sample_index = 0
    clevr_dataset = build_dataset(params)
    # video_sample = clevr_dataset[sample_index]  # already processed by clip
    ori_img = get_img(sample_index, clevr_dataset)
    # print('img.shape = ', img.shape)
    img = Image.fromarray(ori_img)
    raw_text = clevr_dataset._generate_text(sample_index)

    # clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(params.clip_arch, device=device)
    clip_detector = CLIPDetectorV0(model, preprocess, device)

    # get coords and anchor features
    COUNT_W, COUNT_H = 20, 20
    image_w, image_h = img.size
    coords = get_anchor_coords(image_w, image_h, COUNT_W, COUNT_H)
    anchor_features = clip_detector.get_anchor_features(img, coords)
    # print('coords: ', coords)
    print(f'#windows = {len(coords)}')
    print('raw_text = ', raw_text)

    for i, (label, texts, colour, k) in enumerate([
        ('a cyan cylinder', ['a cyan cylinder'], (0, 255, 255), 1),
        ('a brown cylinder', ['a brown cylinder'], (135, 51, 36), 1),
        ('a yellow cube', ['a yellow cube'], (255, 215, 0), 1),
        ('a purple cylinder', ['a purple cylinder'], (128, 128, 255), 1),
    ]):
        result = clip_detector.detect_by_text(
            texts=texts,
            coords=coords,
            anchor_features=anchor_features,
            img=Image.fromarray(ori_img),
            k=k,
            iou_thr=1.0
        )
        img = clip_detector.draw(
            img,
            result,
            label=label,
            colour=colour,
            font_colour=colour,
            font_scale=0.3,
            font_thickness=1,
        )

    plt.figure(num=None, figsize=(128, 128), dpi=300, facecolor='w', edgecolor='k')
    # plt.imshow(img)
    plt.imsave(f'res.png', img)


params = SlidingParams()
main(params)
