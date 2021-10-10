import torch
from pytorch_lightning import Callback

import wandb


class VideoLogCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                video = pl_module.sample_video()
                trainer.logger.experiment.log(
                    {"video": [wandb.Video(video, fps=6)]}, commit=False)
