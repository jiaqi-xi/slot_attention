import importlib
import argparse
from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

from data import CLEVRVideoFrameDataModule
from method import SlotAttentionVideoMethod as SlotAttentionMethod
from utils import VideoLogCallback, ImageLogCallback, rescale
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
            print("INFO: using FP16 training!")

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
    )

    print(
        f"Training set size (images must have {params.num_slots - 1} "
        "objects):", len(clevr_datamodule.train_dataset))

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
        use_relu=params.use_relu,
        slot_mlp_size=params.slot_mlp_size,
        slot_agnostic=params.slot_agnostic,
        random_slot=params.random_slot,
        use_entropy_loss=params.use_entropy_loss,
    )

    method = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params)

    logger_name = f'{args.params}-fp16' if args.fp16 else args.params
    logger = pl_loggers.WandbLogger(
        project="slot-attention-clevr6-video", name=logger_name)

    # saves a file like: 'path/to/ckp/CLEVRVideo001-val_loss=0.0032.ckpt'
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath="./checkpoint/"
        f"{args.params + '-fp16' if args.fp16 else args.params}",
        filename="CLEVRVideo{epoch:03d}-val_loss_{avg_val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

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
    )
    trainer.fit(method, datamodule=clevr_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Slot Attention')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--eval-interval', type=float, default=1.0)
    args = parser.parse_args()
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    params = importlib.import_module(args.params)
    params = params.SlotAttentionParams()
    main(params)
