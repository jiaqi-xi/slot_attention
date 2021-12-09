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
from obj_model import ObjSlotAttentionModel, SemPosSepObjSlotAttentionModel
from obj_params import SlotAttentionParams
from cater_data import ObjCATERVisionLanguageCLIPDataModule

sys.path.append('../')

from train import build_data_transforms, process_ckp
from text_model import ObjMLPText2Slot
from utils import VideoLogCallback, ImageLogCallback

sys.path.append('../viewpoint_dataset/')

from viewpoint_data import ObjCLEVRVisionLanguageViewpointDataModule


def build_data_module(params: SlotAttentionParams):
    if 'viewpoint' in params.data_root:
        data_module = ObjCLEVRVisionLanguageViewpointDataModule
        if '4obj' in params.data_root:
            assert params.num_slots == 5
        else:
            assert params.num_slots == 3
    elif 'CATER' in params.data_root:
        data_module = ObjCATERVisionLanguageCLIPDataModule
        assert params.num_slots == 11
    else:
        data_module = ObjCLEVRVisionLanguageCLIPDataModule
        assert params.num_slots == 7
    clip_transforms = build_data_transforms(params)
    clevr_datamodule = data_module(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clip_transforms=clip_transforms,
        num_workers=params.num_workers,
        max_n_objects=params.num_slots - 1,
        prompt=params.prompt,
        shuffle_obj=params.shuffle_obj,
        pad_text=params.pad_text,
    )
    return clevr_datamodule


def build_text2slot_model(params: SlotAttentionParams):
    if not params.use_text2slot:
        text2slot_model = None
    else:
        text2slot_model = ObjMLPText2Slot(
            params.clip_text_channel,
            params.slot_size,
            params.text2slot_hidden_sizes,
            use_bn=False,
            normalize_slots=params.normalize_slots)
    return text2slot_model


def build_slot_attention_model(params: SlotAttentionParams):
    clip_model, _ = clip.load(params.clip_arch)
    text2slot_model = build_text2slot_model(params)
    print('Using SemPosSepObjSlotAttentionModel!')
    model = SemPosSepObjSlotAttentionModel(
        clip_model=clip_model,
        use_clip_vision=params.use_clip_vision,
        use_clip_text=params.use_text2slot,
        text2slot_model=text2slot_model,
        resolution=params.resolution,
        slot_dict=dict(
            num_slots=params.num_slots,
            num_iterations=params.num_iterations,
            slot_size=params.slot_size,
            slot_mlp_size=params.slot_mlp_size,
            use_bg_sep_slot=params.use_bg_sep_slot,
        ),
        enc_dict=dict(
            out_features=params.out_features,
            kernel_size=params.kernel_size,
            enc_pos_size=params.enc_pos_size,
            use_unet=params.use_unet,
            use_resnet=params.use_resnet,
            enc_channels=params.enc_channels,
            enc_resolution=params.enc_resolution,
            visual_feats_channels=params.clip_vision_channel,
            enc_norm=params.enc_norm if hasattr(params, 'enc_norm') else '',
        ),
        dec_dict=dict(
            dec_pos_size=params.dec_pos_size,
            dec_channels=params.dec_channels,
            dec_resolution=params.dec_resolution,
            dec_norm=params.dec_norm if hasattr(params, 'dec_norm') else '',
        ),
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

    model = build_slot_attention_model(params)

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
    # TODO: save ckp in temp folder '/checkpoint/ziyiwu/xxx/'
    ckp_path = f"/checkpoint/ziyiwu/{SLURM_JOB_ID}"
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
        if ckp_files:
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
