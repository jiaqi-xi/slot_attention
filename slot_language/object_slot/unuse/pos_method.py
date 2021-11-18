import sys
import torch
from torchvision import utils as vutils

from obj_method import ObjSlotAttentionVideoLanguageMethod

sys.path.append('../')

from utils import to_rgb_from_tensor


class ObjPosSlotAttentionVideoLanguageMethod(
        ObjSlotAttentionVideoLanguageMethod):

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = {k: v[idx] for k, v in next(iter(dl)).items()}
        if self.params.gpus > 0:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        recon_combined, recons, masks, all_masks = self.model.forward(batch)

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

        # also visualize all_masks, which is masks from all pos_slots
        # shape is [B, num_slots, num_pos_slot, 1, H, W]
        all_masks = torch.cat([all_masks] * C, dim=3)
        all_masks = all_masks.transpose(2, 1).flatten(0, 1)
        all_masks = vutils.make_grid(
            all_masks.view(-1, C, H, W).cpu(),
            normalize=False,
            nrow=all_masks.shape[1],
        )  # [C, B*H, num_slots*W]

        return images, masks, all_masks
