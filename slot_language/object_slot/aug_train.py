import torch

torch.multiprocessing.set_sharing_strategy('file_system')

import os
import sys
import importlib
import argparse
import numpy as np
from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from obj_train import build_slot_attention_model, build_data_transforms, \
    process_ckp, VideoLogCallback, ImageLogCallback
from obj_data import ObjAugCLEVRVisionLanguageCLIPDataModule
from aug_method import ObjAugSlotAttentionVideoLanguageMethod as SlotAttentionMethod
from aug_model import ObjAugSlotAttentionModel, SemPosSepObjAugSlotAttentionModel
from aug_params import SlotAttentionParams

sys.path.append('../viewpoint_dataset/')

from viewpoint_data import ObjAugCLEVRVisionLanguageViewpointDataModule


def build_data_module(params: SlotAttentionParams):
    if '4obj' in params.data_root:
        assert params.num_slots == 5
    elif 'viewpoint' in params.data_root:
        assert params.num_slots == 3
    else:
        assert params.num_slots == 7
    clip_transforms = build_data_transforms(params)
    data_module = ObjAugCLEVRVisionLanguageViewpointDataModule if 'viewpoint' in \
        params.data_root else ObjAugCLEVRVisionLanguageCLIPDataModule
    clevr_datamodule = data_module(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clip_transforms=clip_transforms,
        num_workers=params.num_workers,
        max_n_objects=params.num_slots - 1,
        shuffle_obj=params.shuffle_obj,
        pad_text=params.pad_text,
        flip_img=params.flip_img,
    )
    return clevr_datamodule


def build_aug_slot_attention_model(params: SlotAttentionParams):
    model = build_slot_attention_model(params)
    model_ = SemPosSepObjAugSlotAttentionModel if \
        params.use_sempos_sep else ObjAugSlotAttentionModel
    model = model_(
        model=model,
        use_contrastive_loss=params.use_contrastive_loss,
        contrastive_T=params.contrastive_T,
        use_text_recon_loss=params.use_text_recon_loss if hasattr(
            params, 'use_text_recon_loss') else False,
        text_recon_mlp=params.text_recon_mlp if hasattr(
            params, 'text_recon_mlp') else (),
        text_recon_normalize=params.text_recon_normalize if hasattr(
            params, 'text_recon_normalize') else False)
    return model


def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(f"INFO: model has {params.num_slots} slots")
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

    model = build_aug_slot_attention_model(params)

    clevr_datamodule = build_data_module(params)

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

    # saves a file like: 'path/to/ckp/CLEVRVideo-001-100000-val=0.0032.ckpt'
    ckp_path = "./checkpoint/" \
        f"{args.params + '-fp16' if args.fp16 else args.params}/{SLURM_JOB_ID}"
    ckp_name = "CLEVRVideo-{epoch:03d}-{step:06d}-val_{val_recon_loss:.4f}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_recon_loss",
        dirpath=ckp_path,
        filename=ckp_name,
        save_top_k=2,
        mode="min",
    )

    # automatically detect previous checkpoint
    # because if SLURM_JOB_ID is equal, that should definitely be the case
    if os.path.exists(ckp_path):
        ckp_files = os.listdir(ckp_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.startswith('CLEVRVideo')]
        step_num = [int(ckp[26:32]) for ckp in ckp_files]
        last_ckp = ckp_files[np.argmax(step_num)]
        print(f'INFO: automatically detect checkpoint {last_ckp}')
        args.weight = os.path.join(ckp_path, last_ckp)

    process_ckp(args.weight)  # enable mid-epoch resuming
    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        gradient_clip_val=params.grad_clip_norm,
        # TODO: 'ddp' doesn't work on Vector cluster!
        accelerator="dp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps
        if not args.weight else 0,
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
        profiler='simple',
        precision=16 if args.fp16 else 32,
        weights_save_path=ckp_path,
    )
    trainer.fit(
        method,
        datamodule=clevr_datamodule,
        ckpt_path=args.weight if args.weight else None)


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
