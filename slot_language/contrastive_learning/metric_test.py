import os
import sys
import importlib
import argparse
import numpy as np
from typing import Optional

import torch
from torchvision import utils as vutils

from pair_data import PairCLEVRVisionLanguageCLIPDataModule
from metric_model import MetricSlotAttentionModel
from metric_method import MetricSlotAttentionVideoLanguageMethod as SlotAttentionMethod
from metric_params import SlotAttentionParams

sys.path.append('../')

from train import build_data_transforms, build_slot_attention_model
from utils import to_rgb_from_tensor, save_video


def build_metric_slot_attention_model(params: SlotAttentionParams):
    model = build_slot_attention_model(params)
    model = MetricSlotAttentionModel(
        model,
        params.slot_size,
        T=params.metric_temperature,
        mlp=params.metric_mlp)
    return model


def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    clip_transforms = build_data_transforms(params)

    model = build_metric_slot_attention_model(params)

    clevr_datamodule = PairCLEVRVisionLanguageCLIPDataModule(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clip_transforms=clip_transforms,
        max_n_objects=params.num_slots - 1,
        num_workers=params.num_workers,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        fine_grained=params.fine_grained,
        object_only=params.object_only,
        overfit=params.overfit,
        separater=params.separater,
    )

    print('Not using max_object_num constraint here!')

    model = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params)

    model.load_state_dict(torch.load(args.weight)['state_dict'], strict=True)
    model = model.cuda().eval()

    save_folder = os.path.join(os.path.dirname(args.weight), 'vis')
    os.makedirs(save_folder, exist_ok=True)

    # get image from train and val dataset
    with torch.no_grad():
        train_res = inference(
            model, clevr_datamodule.train_dataset, num=args.test_num)
        val_res = inference(
            model, clevr_datamodule.val_dataset, num=args.test_num)
    save_video(train_res, os.path.join(save_folder, 'train.mp4'), fps=2)
    save_video(val_res, os.path.join(save_folder, 'val.mp4'), fps=2)


def inference(model, dataset, num=3):
    dataset.is_video = True
    num_data = dataset.num_videos
    data_idx = np.random.choice(num_data, num, replace=False)
    results = []
    all_texts = []
    for idx in data_idx:
        batch = dataset.__getitem__(idx)  # dict with key video, text, raw_text
        video, text, raw_text = \
            batch['video'], batch['text'], batch['raw_text']
        all_texts.append(raw_text)
        batch = dict(img=video.float().cuda(), text=text.cuda())
        recon_combined, recons, masks, slots = model(batch)
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch['img'].unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            ))  # [B (temporal dim), num_slots+2, 3, H, W]

        T, num_slots, C, H, W = recons.shape
        video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                normalize=False,
                nrow=out.shape[1],
            ) for i in range(T)
        ])  # [T, 3, H, (num_slots+2)*W]
        results.append(video.numpy())

    # concat results vertically
    results = np.concatenate(results, axis=2)  # [T, 3, B*H, (num_slots+2)*W]
    results = np.ascontiguousarray(results.transpose((0, 2, 3, 1)))
    return results


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    parser = argparse.ArgumentParser(description='Test Slot Attention')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--test-num', type=int, default=5)
    # TODO: I didn't find improvement using num-iter=5 as stated in the paper
    parser.add_argument('--num-iter', type=int, default=3)
    args = parser.parse_args()
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    params = importlib.import_module(args.params)
    params = params.SlotAttentionParams()
    params.num_iterations = args.num_iter
    main(params)
