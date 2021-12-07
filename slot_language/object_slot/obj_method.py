import sys
import numpy as np
import torch.optim as optim

sys.path.append('../')

from method import SlotAttentionVideoLanguageMethod


class ObjSlotAttentionVideoLanguageMethod(SlotAttentionVideoLanguageMethod):

    def __init__(self, model, datamodule, params):
        super().__init__(model, datamodule, params)

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
