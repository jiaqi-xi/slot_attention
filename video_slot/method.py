import pytorch_lightning as pl
import numpy as np
import torch
from torch import optim
from torchvision import utils as vutils

from model import SlotAttentionModel, ConvAutoEncoder
from video_model import RecurrentSlotAttentionModel
from params import SlotAttentionParams
from utils import Tensor, to_rgb_from_tensor


class SlotAttentionVideoMethod(pl.LightningModule):

    def __init__(self, model: SlotAttentionModel, predictor: ConvAutoEncoder,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.predictor = predictor
        self.datamodule = datamodule
        self.params = params
        self.entropy_loss_w = params.entropy_loss_w

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        assert len(input.shape) == 5, 'invalid model input shape!'  # [B,num,.]
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Compute loss in one training step.
        We need both reconstruction loss and a future prediction loss.
        batch: [B, sample_clip_num, C, H, W]
        """
        train_output = self.model.loss_function(batch)
        loss = train_output['recon_loss']
        train_loss = {
            'recon_loss': train_output['recon_loss'],
        }
        if 'entropy' in train_output.keys():
            loss = loss + train_output['entropy'] * self.entropy_loss_w
            train_loss['entropy'] = train_output['entropy']
        train_loss['loss'] = loss
        logs = {key: val.item() for key, val in train_loss.items()}
        # record training time
        logs['data_time'] = \
            self.trainer.profiler.recorded_durations['get_train_batch'][-1]
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_output = self.model.loss_function(batch)
        val_loss = {'recon_loss': val_output['recon_loss']}
        if 'entropy' in val_output.keys():
            val_loss['entropy'] = val_output['entropy']
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_recon_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        logs = {
            'val_loss': avg_recon_loss,
            'val_recon_loss': avg_recon_loss,
        }
        if self.model.use_entropy_loss:
            avg_entropy = torch.stack([x['entropy'] for x in outputs]).mean()
            logs['val_entropy'] = avg_entropy
            logs['val_loss'] += avg_entropy * self.entropy_loss_w
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = next(iter(dl))[idx]  # [B, sample_clip_num, C, H, W]
        bs, clip, C, H, W = batch.shape
        if self.params.gpus > 0:
            batch = batch.to(self.device)
        # recon_combined: [B*clip, C, H, W]
        # masks: [B*clip, num_slots, 1, H, W]
        # slots: [B*clip, num_slots, C, H, W]
        batch = batch.view(-1, C, H, W)
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            ))  # [B*clip, num_slots+2, C, H, W]

        images = vutils.make_grid(
            out.view(bs * clip * out.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=out.shape[1],
        )  # [C, (B*clip)*H, (num_slots+2)*W]

        # also visualize the mask
        masks = torch.cat([masks] * C, dim=2)  # [B, num_slots, C, H, W]
        masks = vutils.make_grid(
            masks.view(bs * clip * masks.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=masks.shape[1],
        )  # [C, B*H, num_slots*W]

        return images, masks

    def sample_video(self):
        dst = self.datamodule.val_dataset
        dst.is_video = True
        sampled_idx = torch.randperm(dst.num_videos)[:self.params.n_samples]
        results = []
        for idx in sampled_idx:
            idx = idx.item()
            video = dst.__getitem__(idx)  # [num_clips, C, H, W]
            if self.params.gpus > 0:
                video = video.to(self.device)
            output = self.model.forward(video.unsqueeze(0))
            recon_combined, recons, masks, slots = output
            # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
            out = to_rgb_from_tensor(
                torch.cat(
                    [
                        video.unsqueeze(1),  # original images
                        recon_combined.unsqueeze(1),  # reconstructions
                        recons * masks + (1 - masks),  # each slot
                    ],
                    dim=1,
                ))  # [B (temporal dim), num_slots+2, C, H, W]
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
        video = torch.cat(results, dim=2)  # [T, C, B*H, (num_slots+2)*W]

        return video

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
