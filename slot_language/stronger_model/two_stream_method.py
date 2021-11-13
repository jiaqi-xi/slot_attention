import sys
import pytorch_lightning as pl
import torch
from torchvision import utils as vutils

from two_stream_model import TwoStreamSlotAttentionModel
from two_stream_params import SlotAttentionParams

sys.path.append('../')

from utils import to_rgb_from_tensor
from method import SlotAttentionVideoLanguageMethod


class TwoStreamSlotAttentionVideoLanguageMethod(
        SlotAttentionVideoLanguageMethod):

    def __init__(self, model: TwoStreamSlotAttentionModel,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams):
        super().__init__(model, datamodule, params)
        self.entropy_loss_w = params.entropy_loss_w

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = {k: v[idx] for k, v in next(iter(dl)).items()}
        if self.params.gpus > 0:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        recon_combined, recons, masks, coarse_masks = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch['img'].unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            ))  # [B, num_slots+2, C, H, W]

        bs, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(bs * out.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=out.shape[1],
        )  # [C, B*H, (num_slots+2)*W]

        # also visualize the mask of slots
        # masks of shape [B, num_slots, 1, H, W]
        masks = torch.cat([masks] * C, dim=2)  # [B, num_slots, C, H, W]
        masks = vutils.make_grid(
            masks.view(bs * masks.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=masks.shape[1],
        )  # [C, B*H, num_slots*W]

        # the same goes to coarse_mask
        coarse_masks = torch.cat([coarse_masks] * C, dim=2)
        coarse_masks = vutils.make_grid(
            coarse_masks.view(bs * coarse_masks.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=coarse_masks.shape[1],
        )  # [C, B*H, num_slots*W]

        return images, masks, coarse_masks
