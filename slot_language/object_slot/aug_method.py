import sys
import torch
from torchvision import utils as vutils
import pytorch_lightning as pl

from obj_method import ObjSlotAttentionVideoLanguageMethod
from pos_model import ObjPosSlotAttentionModel
from aug_model import ObjAugSlotAttentionModel
from aug_params import SlotAttentionParams

sys.path.append('../')

from utils import to_rgb_from_tensor


class ObjAugSlotAttentionVideoLanguageMethod(
        ObjSlotAttentionVideoLanguageMethod):

    def __init__(self, model: ObjAugSlotAttentionModel,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams):
        super().__init__(model, datamodule, params)
        self.entropy_loss_w = params.entropy_loss_w
        self.equivariance_loss_w = params.equivariance_loss_w

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        loss = train_loss['recon_loss'] + \
            train_loss['equivariance_loss'] * self.equivariance_loss_w
        if 'entropy' in train_loss.keys():
            loss = loss + train_loss['entropy'] * self.entropy_loss_w
        train_loss['loss'] = loss
        logs = {key: val.item() for key, val in train_loss.items()}
        # record training time
        logs['data_time'] = \
            self.trainer.profiler.recorded_durations['get_train_batch'][-1]
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def sample_images(self):
        if not self.params.flip_img:
            return super().sample_images()
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = {k: v[idx] for k, v in next(iter(dl)).items()}
        if self.params.gpus > 0:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        batch = dict(
            img=torch.stack([batch['img'], batch['flipped_img']],
                            dim=1).flatten(0, 1),
            text=torch.stack([batch['text'], batch['text']],
                             dim=1).flatten(0, 1),
            padding=torch.stack([batch['padding'], batch['padding']],
                                dim=1).flatten(0, 1))
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

        if isinstance(self.model.model, ObjPosSlotAttentionModel):
            all_masks = slots
            all_masks = torch.cat([all_masks] * C, dim=3)
            all_masks = all_masks.transpose(2, 1).flatten(0, 1)
            all_masks = vutils.make_grid(
                all_masks.view(-1, C, H, W).cpu(),
                normalize=False,
                nrow=all_masks.shape[1],
            )  # [C, B*H, num_slots*W]
            return images, masks, all_masks

        return images, masks
