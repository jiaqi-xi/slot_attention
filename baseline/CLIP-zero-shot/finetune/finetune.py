import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import clip
from utils import AverageMeter
from params import FinetuneParams
from data import CLEVRVisionLanguageCLIPDataModule


def build_datamodule(params: FinetuneParams, transform):
    datamodule = CLEVRVisionLanguageCLIPDataModule(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.batch_size,
        clip_transforms=transform,
        num_workers=params.num_workers,
        prompt=params.prompt,
        separater=params.separater)
    return datamodule


def build_model(params: FinetuneParams, ckp_path=None):
    model, preprocess = clip.load(params.clip_arch, device=device, jit=False)
    if params.freeze_text_encoder:
        model = clip.freeze_text_encoder(model)
    if device == "cpu":
        model.float()
    else:
        clip.convert_weights(model)

    # Params used from paper
    # the lr is smaller, more safe for finetuning on new dataset
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.2)
    start_epoch = 0

    if ckp_path is not None:
        checkpoint = torch.load(ckp_path)
        checkpoint['model_state_dict'][
            'input_resolution'] = model.input_resolution  # 224
        checkpoint['model_state_dict'][
            'context_length'] = model.context_length  # 77
        checkpoint['model_state_dict']['vocab_size'] = model.vocab_size

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    return model, preprocess, optimizer, start_epoch


def save_ckp(save_name, model, optimizer, epoch):
    ckp = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(ckp, save_name)


def main():
    for epoch in range(start_epoch, params.epochs):
        train_loader = datamodule.train_dataloader()
        loss_avg = AverageMeter()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images, texts = batch['img'].cuda(), batch['text'].cuda()
            B = images.shape[0]

            logits_per_image, logits_per_text = model(
                images, texts, fix_text=params.freeze_text_encoder)

            ground_truth = torch.arange(B, dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) +
                          loss_txt(logits_per_text, ground_truth)) / 2.0
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                clip.convert_models_to_fp32(model)
                optimizer.step()
                clip.convert_weights(model)

            loss_avg.update(total_loss.item(), B)
            if step % 50 == 0:
                print(f'Epoch {epoch}, step {step}, loss: {loss_avg.avg:.6f}')
                wandb.log({'loss': loss_avg.avg})
                loss_avg = AverageMeter()

        save_name = os.path.join(logs_dir, f'clip{epoch}.pth')
        save_ckp(save_name, model, optimizer, epoch)

        with torch.no_grad():
            val_loss = val_epoch()
        print(f'Epoch {epoch}, validation loss: {val_loss:.6f}')


def val_epoch():
    val_loader = datamodule.val_dataloader()
    loss_avg = AverageMeter()
    for batch in val_loader:
        images, texts = batch['img'].cuda(), batch['text'].cuda()
        B = images.shape[0]
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(B, dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_text, ground_truth)) / 2.0
        loss_avg.update(total_loss.item(), B)
    return loss_avg.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune CLIP')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    params = FinetuneParams()
    model, preproc, optimizer, start_epoch = build_model(params, args.weight)
    datamodule = build_datamodule(params, preproc)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # log
    start_datetime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    logs_dir = f"logs/{start_datetime}"
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    exp_name = f'finetune_clip-{SLURM_JOB_ID}'
    os.makedirs(logs_dir, exist_ok=True)
    wandb.init(
        project='slot-attention-clevr6-language-video',
        name=exp_name,
        id=exp_name)

    main()
