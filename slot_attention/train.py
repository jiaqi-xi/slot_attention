import importlib
import argparse
from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

from data import CLEVRDataModule
from method import SlotAttentionMethod
from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import ImageLogCallback
from utils import rescale


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

    clevr_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(rescale),  # rescale between -1 and 1
        # TODO: no center crop
        transforms.Resize(params.resolution),
    ])

    clevr_datamodule = CLEVRDataModule(
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
    )

    method = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params)

    logger_name = args.params
    logger = pl_loggers.WandbLogger(
        project="slot-attention-clevr6", name=logger_name)

    # saves a file like: 'path/to/ckp/CLEVR001-val_loss=0.32.ckpt'
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath=f"./checkpoint/{args.params}",
        filename="CLEVR{epoch:03d}-val_loss_{avg_val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[
            LearningRateMonitor("step"),
            ImageLogCallback(), checkpoint_callback
        ] if params.is_logger_enabled else [checkpoint_callback],
    )
    trainer.fit(method, datamodule=clevr_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Slot Attention')
    parser.add_argument('--params', type=str, default='params')
    args = parser.parse_args()
    params = importlib.import_module(args.params)
    params = params.SlotAttentionParams()
    main(params)
