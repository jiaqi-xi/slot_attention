import os
import sys
import importlib
import argparse
import numpy as np
from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import clip
from obj_data import ObjCLEVRVisionLanguageCLIPDataModule
from obj_method import ObjSlotAttentionVideoLanguageMethod as SlotAttentionMethod
from obj_model import ObjSlotAttentionModel
from obj_params import SlotAttentionParams

sys.path.append('../')

from train import build_data_transforms, process_ckp
from text_model import ObjMLPText2Slot
from utils import VideoLogCallback, ImageLogCallback

sys.path.append('../viewpoint_dataset/')

from viewpoint_data import ObjCLEVRVisionLanguageViewpointDataModule


def build_text2slot_model(params: SlotAttentionParams):
    if not params.use_text2slot:
        text2slot_model = None
    else:
        text2slot_model = ObjMLPText2Slot(
            params.clip_text_channel,
            params.slot_size,
            params.text2slot_hidden_sizes,
            use_bn=False)
    return text2slot_model


def build_slot_attention_model(params: SlotAttentionParams):
    clip_model, _ = clip.load(params.clip_arch)
    text2slot_model = build_text2slot_model(params)
    model = ObjSlotAttentionModel(
        clip_model=clip_model,
        use_clip_vision=params.use_clip_vision,
        use_clip_text=params.use_text2slot,
        text2slot_model=text2slot_model,
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        enc_resolution=params.enc_resolution,
        enc_channels=params.clip_vision_channel,
        enc_pos_enc=params.enc_pos_enc,
        slot_size=params.slot_size,
        dec_kernel_size=params.dec_kernel_size,
        dec_hidden_dims=params.dec_channels,
        dec_resolution=params.dec_resolution,
        slot_mlp_size=params.slot_mlp_size,
        use_entropy_loss=params.use_entropy_loss,
    )
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

    clip_transforms = build_data_transforms(params)

    model = build_slot_attention_model(params)

    data_module = ObjCLEVRVisionLanguageViewpointDataModule if 'viewpoint' in \
        params.data_root else ObjCLEVRVisionLanguageCLIPDataModule
    clevr_datamodule = data_module(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clip_transforms=clip_transforms,
        num_workers=params.num_workers,
        max_n_objects=params.num_slots - 1,
        shuffle_obj=params.shuffle_obj,
    )

    method = SlotAttentionMethod(
        model=model,
        datamodule=clevr_datamodule,
        params=params,
        entropy_loss_w=params.entropy_loss_w)

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
