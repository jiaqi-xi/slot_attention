import sys

sys.path.append('../')

import pytorch_lightning as pl
import torch
from torchvision import utils as vutils

from utils import to_rgb_from_tensor
from method import SlotAttentionVideoLanguageMethod
from contrastive_model import MoCoSlotAttentionModel
from contrastive_params import SlotAttentionParams


class MoCoSlotAttentionVideoLanguageMethod(SlotAttentionVideoLanguageMethod):

    def __init__(self, model: MoCoSlotAttentionModel,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams):
        super().__init__(model, datamodule, params)
        self.entropy_loss_w = params.entropy_loss_w
        self.contrastive_loss_w = params.contrastive_loss_w

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        loss = train_loss['recon_loss'] + \
            self.contrastive_loss_w * train_loss['contrastive_loss']
        if 'entropy' in train_loss.keys():
            loss = loss + train_loss['entropy'] * self.entropy_loss_w
        train_loss['loss'] = loss
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = {k: v[idx] for k, v in next(iter(dl)).items()}
        if self.params.gpus > 0:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        B, C, H, W = batch['img'].shape
        batch = dict(
            img=torch.stack([batch['img'], batch['img2']],
                            dim=1).view(2 * B, C, H, W),
            text=torch.stack([batch['text'], batch['text2']],
                             dim=1).view(2 * B, -1))
        recon_combined, recons, masks, slots = self.model.forward(batch)

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

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=out.shape[1],
        )  # [C, B*H, (num_slots+2)*W]

        # also visualize the mask of slots
        # masks of shape [B, num_slots, 1, H, W]
        masks = torch.cat([masks] * C, dim=2)  # [B, num_slots, C, H, W]
        masks = vutils.make_grid(
            masks.view(batch_size * masks.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=masks.shape[1],
        )  # [C, B*H, num_slots*W]

        return images, masks
