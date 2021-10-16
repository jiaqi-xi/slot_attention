import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from model import SlotAttentionModel, ConvAutoEncoder
from video_model import RecurrentSlotAttentionModel
from params import SlotAttentionParams
from utils import Tensor, to_rgb_from_tensor


class SlotAttentionVideoMethod(pl.LightningModule):

    def __init__(self,
                 model: SlotAttentionModel,
                 predictor: ConvAutoEncoder,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams,
                 pred_mask: bool = True,
                 stop_future_grad: bool = False):
        super().__init__()
        self.model = model
        self.predictor = predictor
        self.datamodule = datamodule
        self.params = params
        # whether use mask as future prediction input?
        self.pred_mask = pred_mask
        # stop grad for GT of future prediction?
        self.stop_future_grad = stop_future_grad

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        assert len(input.shape) == 4, 'invalid model input shape!'  # [B,C,H,W]
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Compute loss in one training step.
        We need both reconstruction loss and a future prediction loss.
        batch: [B, sample_clip_num, C, H, W]
        """
        bs, clip, C, H, W = batch.shape[:]
        train_output = self.model.loss_function(batch.view(-1, C, H, W))
        train_loss = {'loss': train_output['loss']}
        if self.predictor is not None:
            # masks: [B*clip, num_slots, 1, H, W]
            # slots: [B*clip, num_slots, C, H, W]
            slots, masks = train_output['slots'], train_output['masks']
            # get [B*(clip-1)*num_slots, C', H, W] input and gt
            prev_input, future_gt = self._prepare_predictor_data(
                masks if self.pred_mask else slots, bs, clip)
            if self.stop_future_grad:
                future_gt = future_gt.detach().clone()
            pred_loss = self.predictor.loss_function(prev_input, future_gt)
            pred_loss = {'pred_loss': pred_loss['pred_loss']}
            train_loss.update(pred_loss)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        bs, clip, C, H, W = batch.shape[:]
        val_output = self.model.loss_function(batch.view(-1, C, H, W))
        val_loss = {'loss': val_output['loss']}
        if self.predictor is not None:
            # masks: [B*clip, num_slots, 1, H, W]
            # slots: [B*clip, num_slots, C, H, W]
            slots, masks = val_output['slots'], val_output['masks']
            # get [B*(clip-1)*num_slots, C', H, W] input and gt
            prev_input, future_gt = self._prepare_predictor_data(
                masks if self.pred_mask else slots, bs, clip)
            if self.stop_future_grad:
                future_gt = future_gt.detach().clone()
            pred_loss = self.predictor.loss_function(prev_input, future_gt)
            pred_loss = {'pred_loss': pred_loss['pred_loss']}
            val_loss.update(pred_loss)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        if self.predictor is not None:
            avg_pred_loss = torch.stack([x["pred_loss"]
                                         for x in outputs]).mean()
            logs['avg_val_pred_loss'] = avg_pred_loss
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def _prepare_predictor_data(self, data, bs, clip):
        """data in shape [B*clip, num_slots, C', H, W]"""
        _, num_slots, C, H, W = data.shape[:]
        data = data.view(bs, clip, num_slots, C, H, W)
        prev_input = data[:, :-1].reshape(bs * (clip - 1) * num_slots, C, H, W)
        future_gt = data[:, 1:].reshape(bs * (clip - 1) * num_slots, C, H, W)
        return prev_input, future_gt

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[:self.params.n_samples]
        batch = next(iter(dl))[idx]  # [B, sample_clip_num, C, H, W]
        bs, clip, C, H, W = batch.shape[:]
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

        num_slots = slots.shape[1]
        images = vutils.make_grid(
            out.view(bs * clip * out.shape[1], C, H, W).cpu(),
            normalize=False,
            nrow=out.shape[1],
        )  # [C, (B*clip)*H, (num_slots+2)*W]

        # visualize pred future and real future
        # get [B*(clip-1)*num_slots, C', H, W] input and gt
        prev_input, future_gt = self._prepare_predictor_data(
            masks if self.pred_mask else slots, bs, clip)
        if self.predictor is not None:
            pred_future = self.predictor.forward(prev_input)
        else:
            pred_future = prev_input
        future_gt = future_gt.view(bs * (clip - 1), num_slots, -1, H, W)
        pred_future = pred_future.view(bs * (clip - 1), num_slots, -1, H, W)
        pred_gt = torch.cat(
            [rv for r in zip(pred_future, future_gt) for rv in r], dim=0)
        if self.pred_mask:  # gray to RGB
            pred_gt = torch.cat(
                [
                    pred_gt,
                ] * 3,
                dim=2,
            )  # [2*B*(clip-1), num_slots, C, H, W]
        else:
            pred_gt = to_rgb_from_tensor(pred_gt)
        pred_gt_grid = vutils.make_grid(
            pred_gt.view(2 * bs * (clip - 1) * num_slots, 3, H, W).cpu(),
            normalize=False,
            nrow=num_slots,
        )  # [C, 2*B*(clip-1)*H, num_slots*W]

        return images, pred_gt_grid

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
            if isinstance(self.model, RecurrentSlotAttentionModel):
                output = self.model.forward(video, num_clips=video.shape[0])
            else:
                output = self.model.forward(video)
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
