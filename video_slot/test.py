import os
import importlib
import argparse
import numpy as np

import torch
from torchvision import utils as vutils

from train import build_model, build_datamodule
from method import SlotAttentionVideoMethod as SlotAttentionMethod
from video_model import RecurrentSlotAttentionModel
from params import SlotAttentionParams
from utils import to_rgb_from_tensor, save_video


def main(params=None):
    if params is None:
        params = SlotAttentionParams()

    model = build_model(params)
    clevr_datamodule = build_datamodule(params)

    model = SlotAttentionMethod(
        model=model,
        predictor=None,
        datamodule=clevr_datamodule,
        params=params)
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
    for idx in data_idx:
        video = dataset.__getitem__(idx).float().cuda()
        output = model(video.unsqueeze(0))
        recon_combined, recons, masks, slots = output
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # original images
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
