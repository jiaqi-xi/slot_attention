import sys
import torch
from torchvision import utils as vutils

from obj_method import ObjSlotAttentionVideoLanguageMethod

sys.path.append('../')

from utils import to_rgb_from_tensor


class ObjAugSlotAttentionVideoLanguageMethod(
        ObjSlotAttentionVideoLanguageMethod):

    def sample_images(self):
        if not self.params.flip_img:
            return super().sample_images()
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        data = next(iter(dl))
        batch = {}
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                batch[k] = {k_: v_[idx] for k_, v_ in v.items()}
            else:
                batch[k] = v[idx]
        stack_img = torch.stack([batch['img'], batch['flipped_img']],
                                dim=1).flatten(0, 1).to(self.device)
        text = batch['text']
        if not isinstance(text, torch.Tensor):
            stack_text = {
                k: torch.stack([v, v], dim=1).flatten(0, 1).to(self.device)
                for k, v in text.items()
            }
        else:
            stack_text = torch.stack([text, text],
                                     dim=1).flatten(0, 1).to(self.device)
        batch = dict(img=stack_img, text=stack_text)
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch['img'].unsqueeze(1).type_as(recons),  # ori images
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
