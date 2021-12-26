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
from vision_model import CLIPVisionEncoder
from text_model import MLPText2Slot, TransformerText2Slot, \
    CLIPTextEncoder, TransformerTextEncoder
from detr_module import DETRText2Slot
from data import CLEVRVisionLanguageCLIPDataModule
from method import SlotAttentionVideoLanguageMethod as SlotAttentionMethod
from utils import VideoLogCallback, ImageLogCallback, simple_rescale
from model import SlotAttentionModel
from params import SlotAttentionParams


def build_data_transforms(params: SlotAttentionParams):
    _, clip_transforms = clip.load(params.clip_arch)
    if not params.use_clip_vision:
        from torchvision.transforms import Compose, Resize, ToTensor, \
            Normalize, Lambda, CenterCrop, RandomCrop
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        normalize = Lambda(
            simple_rescale) if params.simple_normalize else Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711))
        transforms = [
            Resize(params.resolution, interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            normalize,
        ]
        if hasattr(params, 'center_crop') and params.center_crop is not None:
            transforms.insert(0, CenterCrop(params.center_crop))
        elif hasattr(params, 'random_crop') and params.random_crop is not None:
            transforms.insert(0, RandomCrop(params.random_crop))
        clip_transforms = Compose(transforms)
    return clip_transforms


def build_data_module(params: SlotAttentionParams):
    clip_transforms = build_data_transforms(params)
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
    return clevr_datamodule


def build_text2slot_model(params: SlotAttentionParams):
    if not params.use_text2slot:
        text2slot_model = None
    elif params.text2slot_arch == 'MLP':
        text2slot_model = MLPText2Slot(
            params.clip_text_channel,
            params.num_slots,
            params.slot_size,
            params.text2slot_hidden_sizes,
            predict_dist=params.predict_slot_dist,
            use_bn=False)
    else:
        if params.text2slot_arch == 'Transformer':
            Text2Slot = TransformerText2Slot
        else:
            Text2Slot = DETRText2Slot
        text2slot_model = Text2Slot(
            params.clip_text_channel,
            params.num_slots,
            params.slot_size,
            d_model=params.text2slot_hidden,
            nhead=params.text2slot_nhead,
            num_layers=params.text2slot_num_transformers,
            dim_feedforward=params.text2slot_dim_feedforward,
            dropout=params.text2slot_dropout,
            activation=params.text2slot_activation,
            text_pe=params.text2slot_text_pe,
            out_mlp_layers=params.text2slot_mlp_layers)
    return text2slot_model


def build_text_encoder(params: SlotAttentionParams, clip_model):
    text_encoder = params.text_encoder if \
        hasattr(params, 'text_encoder') else 'clip'
    if not text_encoder:
        return None
    context_len = params.context_len if hasattr(params, 'context_len') else 0
    if text_encoder == 'clip':
        text_encoder = CLIPTextEncoder(clip_model, context_len=context_len)
    else:
        text_encoder = TransformerTextEncoder(
            text_encoder, context_len=context_len)
    return text_encoder


def build_vision_encoder(params: SlotAttentionParams, clip_model):
    if not params.use_clip_vision:
        return None
    return CLIPVisionEncoder(clip_model)


def build_slot_attention_model(params: SlotAttentionParams):
    clip_model, _ = clip.load(params.clip_arch)
    vision_encoder = build_vision_encoder(params, clip_model)
    text_encoder = build_text_encoder(params, clip_model)
    text2slot_model = build_text2slot_model(params)
    model = SlotAttentionModel(
        clip_vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        text2slot_model=text2slot_model,
        resolution=params.resolution,
        slot_dict=dict(
            num_slots=params.num_slots,
            num_iterations=params.num_iterations,
            slot_size=params.slot_size,
            slot_mlp_size=params.slot_mlp_size,
            use_bg_sep_slot=params.use_bg_sep_slot),
        enc_dict=dict(
            out_features=params.out_features,
            kernel_size=params.kernel_size,
            enc_channels=params.enc_channels,
            enc_resolution=params.enc_resolution,
            visual_feats_channels=params.clip_vision_channel,
        ),
        dec_dict=dict(
            dec_channels=params.dec_channels,
            dec_resolution=params.dec_resolution,
        ),
        use_entropy_loss=params.use_entropy_loss,
        use_word_set=params.use_text2slot
        and params.text2slot_arch in ['Transformer', 'DETR'],
        use_padding_mask=params.use_text2slot
        and params.text2slot_arch in ['Transformer', 'DETR']
        and params.text2slot_padding_mask,
    )
    return model


def process_ckp(ckp_path):
    """Hack that enables checkpointing from mid-epoch."""
    if not ckp_path:
        return
    ckp = torch.load(ckp_path, map_location='cpu')
    for key in ckp['loops']['fit_loop'][
            'epoch_loop.val_loop.dataloader_progress']['current'].keys():
        ckp['loops']['fit_loop']['epoch_loop.val_loop.dataloader_progress'][
            'total'][key] += ckp['loops']['fit_loop'][
                'epoch_loop.val_loop.dataloader_progress']['current'][key]
        ckp['loops']['fit_loop']['epoch_loop.val_loop.dataloader_progress'][
            'current'][key] = 0
    for key in ckp['loops']['fit_loop'][
            'epoch_loop.val_loop.epoch_loop.batch_progress']['current'].keys():
        ckp['loops']['fit_loop'][
            'epoch_loop.val_loop.epoch_loop.batch_progress']['total'][
                key] += ckp['loops']['fit_loop'][
                    'epoch_loop.val_loop.epoch_loop.batch_progress'][
                        'current'][key]
        ckp['loops']['fit_loop'][
            'epoch_loop.val_loop.epoch_loop.batch_progress']['current'][
                key] = 0
    torch.save(ckp, ckp_path)


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
