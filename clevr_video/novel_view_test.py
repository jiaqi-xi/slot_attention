"""Test slot-attention on novel view image pairs"""

import os
import importlib
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision import utils as vutils

from novel_view_data import CLEVRNovelViewImagePairDataset
from method import SlotAttentionVideoMethod as SlotAttentionMethod
from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import rescale, to_rgb_from_tensor


def main(params=None):
    if params is None:
        params = SlotAttentionParams()

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
        use_relu=params.use_relu,
        slot_mlp_size=params.slot_mlp_size,
        learnable_slot=params.learnable_slot,
        slot_agnostic=params.slot_agnostic,
        random_slot=params.random_slot,
    )

    clevr_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(rescale),  # rescale between -1 and 1
        transforms.Resize(params.resolution),
    ])

    test_dst = CLEVRNovelViewImagePairDataset(args.data_root, clevr_transforms)

    model = SlotAttentionMethod(model=model, datamodule=None, params=params)
    model.load_state_dict(torch.load(args.weight)['state_dict'], strict=True)
    model = model.cuda().eval()

    save_folder = os.path.join(os.path.dirname(args.weight), 'vis')
    os.makedirs(save_folder, exist_ok=True)

    # get image from train and val dataset
    with torch.no_grad():
        test_res = inference(model, test_dst, num=args.test_num)
    test_res.save(os.path.join(save_folder, 'test.png'))


def inference(model, dataset, num=5):
    num_data = len(dataset)
    data_idx = np.random.choice(num_data, num, replace=False)
    results = []
    for idx in data_idx:
        img = dataset.__getitem__(idx).float().cuda()
        recon_combined, recons, masks, slots = model(img)
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    img.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            ))  # [B (2), num_slots+2, 3, H, W]

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=out.shape[1],
        )
        results.append(images)

    # concat results vertically
    results = [transforms.ToPILImage()(image) for image in results]
    results = np.concatenate([np.array(image) for image in results], axis=0)
    results = Image.fromarray(results.astype(np.uint8))
    return results


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    parser = argparse.ArgumentParser(description='Train Slot Attention')
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
