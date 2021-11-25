import sys
import numpy as np
import torch
import torch.optim as optim
from torchvision import utils as vutils

sys.path.append('../')

from method import SlotAttentionVideoLanguageMethod
from utils import to_rgb_from_tensor


class ObjSlotAttentionVideoLanguageMethod(SlotAttentionVideoLanguageMethod):

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
            batch = dict(img=video, text=text, padding=padding)
            if self.params.gpus > 0:
                batch = {k: v.to(self.device) for k, v in batch.items()}
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

        if hasattr(self.params, 'use_slot_pos_emb') and self.params.use_slot_pos_emb:
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

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(
            self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step <= total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            if self.params.cosine_decay:
                factor *= ((np.cos(step / total_steps * np.pi) + 1.) / 2.)
            else:
                factor *= (self.params.scheduler_gamma**(step / decay_steps))
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{
                "scheduler": scheduler,
                "interval": "step",
            }],
        )
