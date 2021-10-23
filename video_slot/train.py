import os
import importlib
import argparse
import numpy as np
from typing import Optional

from torchvision import transforms
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data import CLEVRVideoFrameDataModule
from method import SlotAttentionVideoMethod as SlotAttentionMethod
from utils import VideoLogCallback, ImageLogCallback, rescale
from model import SlotAttentionModel, ConvAutoEncoder, PerceptualLoss
from video_model import RecurrentSlotAttentionModel
from params import SlotAttentionParams


def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print("INFO: limiting the dataset to only images with "
              f"`num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_images:
            print("INFO: restricting the train dataset size to "
                  f"`num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print("INFO: restricting the validation dataset size to "
                  f"`num_val_images`: {params.num_val_images}")
        if args.fp16:
            print('INFO: using FP16 training!')
        if args.weight:
            print(f'INFO: loading checkpoint {args.weight}')

    clevr_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(rescale),  # rescale between -1 and 1
        # TODO: no center crop
        transforms.Resize(params.resolution),
    ])

    clevr_datamodule = CLEVRVideoFrameDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
        sample_clip_num=params.sample_clip_num,
    )

    print(
        f"Training set size (images must have {params.num_slots - 1} "
        "objects):", len(clevr_datamodule.train_dataset))

    if params.recurrent_slot_attention:
        model = RecurrentSlotAttentionModel(
            resolution=params.resolution,
            num_slots=params.num_slots,
            num_iterations=params.num_iterations,
            num_clips=params.sample_clip_num,
            kernel_size=params.kernel_size,
            slot_size=params.slot_size,
            hidden_dims=params.hidden_dims,
            decoder_resolution=params.decoder_resolution,
            use_deconv=params.use_deconv,
            empty_cache=params.empty_cache,
            slot_mlp_size=params.slot_mlp_size,
            learnable_slot=params.learnable_slot,
            stop_recur_slot_grad=params.stop_recur_slot_grad,
            use_entropy_loss=params.use_entropy_loss,
        )
    else:
        model = SlotAttentionModel(
            resolution=params.resolution,
            num_slots=params.num_slots,
            num_iterations=params.num_iterations,
            kernel_size=params.kernel_size,
            slot_size=params.slot_size,
            hidden_dims=params.hidden_dims,
            decoder_resolution=params.decoder_resolution,
            use_deconv=params.use_deconv,
            empty_cache=params.empty_cache,
            slot_mlp_size=params.slot_mlp_size,
            learnable_slot=params.learnable_slot,
            use_entropy_loss=params.use_entropy_loss,
        )
    if params.perceptual_loss:
        predictor = PerceptualLoss(params.perceptual_loss)
    else:
        '''
        predictor = ConvAutoEncoder(
            in_channels=1 if params.pred_mask else 3,
            num_slots=params.num_slots,
            use_softmax=params.pred_mask)
        '''
        predictor = None

    method = SlotAttentionMethod(
        model=model,
        predictor=predictor,
        datamodule=clevr_datamodule,
        params=params,
        entropy_loss_w=params.entropy_loss_w,
        pred_mask=params.pred_mask,
        stop_future_grad=params.stop_future_grad)

    # we want to also resume wandb log if restoring from previous training
    logger_name = f'{args.params}-fp16' if args.fp16 else args.params
    if SLURM_JOB_ID:
        logger_name = f'{logger_name}-{SLURM_JOB_ID}'
    logger = pl_loggers.WandbLogger(
        project="slot-attention-clevr6-video-seq",
        name=logger_name,
        id=logger_name)  # we assume only run one exp per one params setting

    # saves a file like: 'path/to/ckp/CLEVRVideoSlot001-val_loss=0.0032.ckpt'
    ckp_path = "./checkpoint/" \
        f"{args.params + '-fp16' if args.fp16 else args.params}/{SLURM_JOB_ID}"
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath=ckp_path,
        filename="CLEVRVideoSlot{epoch:03d}-val_loss_{avg_val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    # automatically detect previous checkpoint
    # because if SLURM_JOB_ID is equal, that should definitely be the case
    if os.path.exists(ckp_path):
        ckp_files = os.listdir(ckp_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.startswith('CLEVRVideo')]
        epoch_num = [int(ckp[20:23]) for ckp in ckp_files]
        last_ckp = ckp_files[np.argmax(epoch_num)]
        print(f'INFO: automatically detect checkpoint {last_ckp}')
        args.weight = os.path.join(ckp_path, last_ckp)

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        # TODO: 'ddp' doesn't work on Vector cluster!
        accelerator="dp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        val_check_interval=args.eval_interval,
        callbacks=[
            LearningRateMonitor("step"),
            ImageLogCallback(),
            VideoLogCallback(),
            checkpoint_callback,
        ] if params.is_logger_enabled else [checkpoint_callback],
        precision=16 if args.fp16 else 32,
        resume_from_checkpoint=args.weight if args.weight else None,
    )
    trainer.fit(method, datamodule=clevr_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Slot Attention')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--sbatch', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--eval-interval', type=float, default=1.0)
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()
    if args.sbatch:
        assert os.environ.get('SLURM_JOB_ID') is not None, \
            'program not running in sbatch mode!'
        SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    else:
        SLURM_JOB_ID = ''
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    params = importlib.import_module(args.params)
    params = params.SlotAttentionParams()
    if params.perceptual_loss:
        assert not params.pred_mask, \
            'Perceptual loss cannot be applied to mask! Only to slot images.'
    main(params)
