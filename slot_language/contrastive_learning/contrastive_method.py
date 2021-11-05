import sys

sys.path.append('../')

import pytorch_lightning as pl
import torch

from .contrastive_model import MoCoSlotAttentionModel
from method import SlotAttentionVideoLanguageMethod
from params import SlotAttentionParams


class MoCoSlotAttentionVideoLanguageMethod(SlotAttentionVideoLanguageMethod):

    def __init__(self, model: MoCoSlotAttentionModel,
                 datamodule: pl.LightningDataModule,
                 params: SlotAttentionParams):
        super().__init__(model, datamodule, params)
        self.model = model
        self.datamodule = datamodule
        self.params = params
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

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_recon_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        logs = {
            'val_loss': avg_recon_loss,
            'val_recon_loss': avg_recon_loss,
        }
        if self.model.model_q.use_entropy_loss:
            avg_entropy = torch.stack([x['entropy'] for x in outputs]).mean()
            logs['val_entropy'] = avg_entropy
            logs['val_loss'] += avg_entropy * self.entropy_loss_w
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
