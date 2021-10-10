import os
import importlib
import argparse
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
from torchvision import utils as vutils

from data import CLEVRDataModule
from method import SlotAttentionMethod
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
    )

    clevr_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(rescale),  # rescale between -1 and 1
        transforms.Resize(params.resolution),
    ])

    clevr_datamodule = CLEVRDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

    model = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params)
    model.load_state_dict(torch.load(args.weight), strict=True)
    model = model.cuda().eval()

    save_folder = os.path.join(os.path.dirname(args.weight), 'vis')
    os.makedirs(save_folder, exist_ok=True)

    # get image from train and val dataset
    inference(
        model,
        clevr_datamodule.train_dataset,
        save_folder=os.path.join(save_folder, 'train'),
        num=args.test_num)
    inference(
        model,
        clevr_datamodule.val_dataset,
        save_folder=os.path.join(save_folder, 'val'),
        num=args.test_num)


def inference(model, dataset, save_folder, num=10):
    num_data = len(dataset)
    data_idx = np.random.choice(num_data, num, replace=False)
    for idx in data_idx:
        img = dataset.__getitem__(idx).unsqueeze(0).float().cuda()
        recon_combined, recons, masks, slots = model(img)
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    img.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            ))

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=out.shape[1],
        )

        # save result
        new = transforms.ToPILImage()(images)
        new.save(os.path.join(save_folder, f'{idx:06d}.png'))


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    parser = argparse.ArgumentParser(description='Train Slot Attention')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--test-num', type=int, default=10)
    args = parser.parse_args()
    params = importlib.import_module(args.params)
    params = params.SlotAttentionParams()
    main(params)
