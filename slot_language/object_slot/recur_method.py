import sys
import torch
from torchvision import utils as vutils

from obj_method import ObjSlotAttentionVideoLanguageMethod

sys.path.append('../')

from utils import to_rgb_from_tensor


class ObjRecurSlotAttentionVideoLanguageMethod(
        ObjSlotAttentionVideoLanguageMethod):

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = {k: v[idx] for k, v in next(iter(dl)).items()}
        if self.params.gpus > 0:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch['img'].flatten(0, 1).unsqueeze(1),  # original images
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

    def sample_video(self):
        dst = self.datamodule.val_dataset
        dst.is_video = True  # load entire video
        sampled_idx = torch.randperm(dst.num_videos)[:self.params.n_samples]
        results = []
        all_texts = []
        for idx in sampled_idx:
            idx = idx.item()
            batch = dst.__getitem__(idx)  # dict with key video, text, raw_text
            video, text, padding, raw_text = batch['video'], \
                batch['text'], batch['padding'], batch['raw_text']
            all_texts.append(raw_text)
            batch = dict(img=video.unsqueeze(0), text=text, padding=padding)
            if self.params.gpus > 0:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            recon_combined, recons, masks, slots = self.model.forward(batch)
            # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
            out = to_rgb_from_tensor(
                torch.cat(
                    [
                        batch['img'][0].unsqueeze(1),  # original images
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
            results.append(video)

        dst.is_video = False
        video = torch.cat(results, dim=2)  # [T, 3, B*H, (num_slots+2)*W]
        text = '\n'.join(all_texts)

        return video, text
