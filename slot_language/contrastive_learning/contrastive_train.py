import os
import sys
import importlib
import argparse
import numpy as np
from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

sys.path.append('../')

from train import build_data_transforms, build_slot_attention_model
from utils import VideoLogCallback, ImageLogCallback
from .contrastive_data import CLEVRVisionLanguageCLIPDataModule
from .contrastive_model import MoCoSlotAttentionModel
from .contrastive_method import MoCoSlotAttentionVideoLanguageMethod as SlotAttentionMethod
from .contrastive_params import SlotAttentionParams


def build_moco_slot_attention_model(params: SlotAttentionParams):
    model_q = build_slot_attention_model(params)
    model_k = build_slot_attention_model(params)
    model = MoCoSlotAttentionModel(
        model_q,
        model_k,
        params.slot_size,
        K=params.moco_query_size,
        m=params.moco_momentum,
        T=params.moco_temperature)
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

    model = build_moco_slot_attention_model(params)

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
        object_only=params.object_only,
        overfit=params.overfit,
        separater=params.separater,
    )

    print('Not using max_object_num constraint here!')

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
        monitor="val_recon_loss",
        dirpath=ckp_path,
        filename="CLEVRVideo{epoch:03d}-val_loss_{val_recon_loss:.4f}",
        save_top_k=2,
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
