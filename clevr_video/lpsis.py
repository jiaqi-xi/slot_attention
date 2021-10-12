import numpy as np
import lpips

import torch
from torchvision import transforms
from torchvision import utils as vutils

from data import CLEVRVideoFrameDataModule
from method import SlotAttentionVideoMethod as SlotAttentionMethod
from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import rescale, to_rgb_from_tensor

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

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

clevr_datamodule = CLEVRVideoFrameDataModule(
    data_root=params.data_root,
    max_n_objects=params.num_slots - 1,
    train_batch_size=params.batch_size,
    val_batch_size=params.val_batch_size,
    clevr_transforms=clevr_transforms,
    num_train_images=params.num_train_images,
    num_val_images=params.num_val_images,
    num_workers=params.num_workers,
)
train_dst, val_dst = clevr_datamodule.train_dataset, clevr_datamodule.val_dataset

model = SlotAttentionMethod(
    model=model, datamodule=clevr_datamodule, params=params)
# model.load_state_dict(torch.load(args.weight)['state_dict'], strict=True)
model = model.cuda().eval()

# Perceptual loss
loss_fn_vgg = lpips.LPIPS(net='vgg')


def inference(model, dataset, idx):
    dataset.is_video = True
    video = dataset.__getitem__(idx).float().cuda()
    recon_combined, recons, masks, slots = model(video)
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
    images = vutils.make_grid(
        out.view(T * out.shape[1], C, H, W).cpu(),
        normalize=False,
        nrow=out.shape[1])
    images = transforms.ToPILImage()(images)

    # recons and masks: [B, num_slots, C, H, W]
    return images, recons, masks


def vgg_sim(x, y):
    assert -1. <= x.min().item() <= 1.
    assert -1. <= y.min().item() <= 1.
    assert len(x.shape) == 4 and len(y.shape) == 4
    return loss_fn_vgg(x, y)
