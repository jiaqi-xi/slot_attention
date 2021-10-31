import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import Tensor, to_rgb_from_tensor


class SlotAttentionVideoMethod(pl.LightningModule):

    def __init__(self,
                 model: SlotAttentionModel,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams,
                 entropy_loss_w: float = 0.0):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.entropy_loss_w = entropy_loss_w

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        loss = train_loss['recon_loss']
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
        batch = next(iter(dl))[idx]
        if self.params.gpus > 0:
            batch = batch.to(self.device)
        if len(batch.shape) == 5:
            # TODO: for the novel view image dataset
            _, _, C, H, W = batch.shape
            batch = batch.reshape(-1, C, H, W)
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
        dst.is_video = True
        # TODO: since video has different length, we only test one here
        sampled_idx = torch.randperm(dst.num_videos)[:1]
        # sampled_idx = torch.randperm(dst.num_videos)[:self.params.n_samples]
        results = []
        for idx in sampled_idx:
            idx = idx.item()
            video = dst.__getitem__(idx)  # [B, 3, H, W]
            if self.params.gpus > 0:
                video = video.to(self.device)
            recon_combined, recons, masks, slots = self.model.forward(video)
            # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
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

        return video

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        if len(batch.shape) == 5:
            # TODO: for the novel view image dataset
            _, _, C, H, W = batch.shape
            batch = batch.reshape(-1, C, H, W)
        val_loss = self.model.loss_function(batch)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
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
            factor *= self.params.scheduler_gamma**(step / decay_steps)
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
