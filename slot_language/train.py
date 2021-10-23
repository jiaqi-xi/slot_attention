import os
import importlib
import argparse
import numpy as np
from typing import Optional

import torch
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import clip
from text_model import Text2Slot
from data import CLEVRVisionLanguageCLIPDataModule
from method import SlotAttentionVideoLanguageMethod as SlotAttentionMethod
from utils import VideoLogCallback, ImageLogCallback
from model import SlotAttentionModel
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

    # load pre-trained CLIP model
    clip_model, clip_transforms = clip.load(params.clip_arch)
    text2slot_model = Text2Slot(
        params.clip_text_channel,
        params.num_slots,
        params.slot_size,
        params.text2slot_hidden_sizes,
        predict_dist=params.predict_slot_dist,
        use_bn=False)

    model = SlotAttentionModel(
        clip_model=clip_model,
        text2slot_model=text2slot_model,
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        enc_resolution=params.enc_resolution,
        enc_channels=params.clip_vision_channel,
        enc_global_feats=params.clip_global_feats,
        enc_pos_enc=params.enc_pos_enc,
        slot_size=params.slot_size,
        dec_kernel_size=params.dec_kernel_size,
        dec_hidden_dims=params.dec_channels,
        dec_resolution=params.dec_resolution,
        empty_cache=params.empty_cache,
        slot_mlp_size=params.slot_mlp_size,
        use_entropy_loss=params.use_entropy_loss,
    )

    clevr_datamodule = CLEVRVisionLanguageCLIPDataModule(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clip_transforms=clip_transforms,
        max_n_objects=params.num_slots - 1,
        num_workers=params.num_workers,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        fine_grained=params.fine_grained,
    )

    print(
        f"Training set size (images must have {params.num_slots - 1} "
        "objects):", len(clevr_datamodule.train_dataset))

    method = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params)

    # we want to also resume wandb log if restoring from previous training
    logger_name = f'{args.params}-fp16' if args.fp16 else args.params
    if SLURM_JOB_ID:
        logger_name = f'{logger_name}-{SLURM_JOB_ID}'
    logger = pl_loggers.WandbLogger(
        project="slot-attention-clevr6-language-video",
        name=logger_name,
        id=logger_name)  # we assume only run one exp per one params setting

    # saves a file like: 'path/to/ckp/CLEVRVideo001-val_loss=0.0032.ckpt'
    ckp_path = "./checkpoint/" \
        f"{args.params + '-fp16' if args.fp16 else args.params}/{SLURM_JOB_ID}"
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath=ckp_path,
        filename="CLEVRVideo{epoch:03d}-val_loss_{avg_val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    # automatically detect previous checkpoint
    # because if SLURM_JOB_ID is equal, that should definitely be the case
    if os.path.exists(ckp_path):
        ckp_files = os.listdir(ckp_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.startswith('CLEVRVideo')]
        epoch_num = [int(ckp[16:19]) for ckp in ckp_files]
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
    main(params)
