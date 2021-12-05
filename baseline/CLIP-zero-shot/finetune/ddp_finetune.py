import os
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import wandb

import clip
from utils import AverageMeter, all_gather, gather_scalar
from params import FinetuneParams
from data import DDPCLEVRVisionLanguageCLIPDataModule


def build_datamodule(params: FinetuneParams, transform):
    datamodule = DDPCLEVRVisionLanguageCLIPDataModule(
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
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], find_unused_parameters=True)
    if not args.fp32:
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

    if ckp_path:
        checkpoint = torch.load(ckp_path)
        checkpoint['model_state_dict'][
            'input_resolution'] = model.input_resolution  # 224
        checkpoint['model_state_dict'][
            'context_length'] = model.context_length  # 77
        checkpoint['model_state_dict']['vocab_size'] = model.vocab_size

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    return model, preprocess, optimizer, start_epoch


def save_ckp(save_name, model, optimizer, epoch):
    ckp = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(ckp, save_name)


def main():
    global model, optimizer, datamodule, args
    for epoch in range(start_epoch, params.epochs):
        train_loader = datamodule.train_dataloader()
        loss_avg, t_avg = AverageMeter(), AverageMeter()
        model.train()
        for step, batch in enumerate(train_loader):
            start_t = time.time()
            optimizer.zero_grad()
            images = batch['img'].cuda(non_blocking=True)
            texts = batch['text'].cuda(non_blocking=True)
            B = images.shape[0]

            img_feats, scaled_text_feats = model(
                images, texts, fix_text=params.freeze_text_encoder, ddp=True)
            all_img_feats = all_gather(img_feats)
            all_scaled_text_feats = all_gather(scaled_text_feats)
            logits_per_image = img_feats @ all_scaled_text_feats.t()
            logits_per_text = scaled_text_feats @ all_img_feats.t()

            ground_truth = torch.arange(
                B, dtype=torch.long, device=device) + B * args.local_rank

            total_loss = (loss_img(logits_per_image, ground_truth) +
                          loss_txt(logits_per_text, ground_truth)) / 2.0

            total_loss.backward()
            if not args.fp32:
                clip.convert_models_to_fp32(model)
                optimizer.step()
                clip.convert_weights(model)
            else:
                optimizer.step()

            if args.local_rank == 0:
                # total_loss = gather_scalar(total_loss).mean()
                loss_avg.update(total_loss.item(), B)
                t_avg.update(time.time() - start_t)

            if args.local_rank == 0 and step % 50 == 0:
                gpu_used = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
                print(f'Epoch {epoch}, step {step}, loss: {loss_avg.avg:.6f}\n'
                      f'Time: {t_avg.avg:.2f}s, GPU memory {gpu_used:.2f}GB')
                wandb.log({'loss': loss_avg.avg, 'time': t_avg.avg})
                loss_avg, t_avg = AverageMeter(), AverageMeter()
            torch.distributed.barrier()

        save_name = os.path.join(logs_dir, f'clip{epoch}.pth')
        if args.local_rank == 0:
            save_ckp(save_name, model, optimizer, epoch)

        with torch.no_grad():
            try:
                val_loss = val_epoch(model.module, datamodule)
            except:
                val_loss = torch.zeros([])

        if args.local_rank == 0:
            print(f'Epoch {epoch}, validation loss: {val_loss:.6f}')
            wandb.log({'val_loss': val_loss}, step=epoch)
        torch.distributed.barrier()


def val_epoch(model, datamodule):
    val_loader = datamodule.val_dataloader()
    loss_avg = AverageMeter()
    model.eval()
    for batch in tqdm(val_loader):
        images = batch['img'].cuda(non_blocking=True)
        texts = batch['text'].cuda(non_blocking=True)
        B = images.shape[0]

        img_feats, scaled_text_feats = model(images, texts, ddp=True)
        all_img_feats = all_gather(img_feats)
        all_scaled_text_feats = all_gather(scaled_text_feats)
        logits_per_image = img_feats @ all_scaled_text_feats.t()
        logits_per_text = scaled_text_feats @ all_img_feats.t()
        ground_truth = torch.arange(
            B, dtype=torch.long, device=device) + B * args.local_rank
        total_loss = (loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_text, ground_truth)) / 2.0

        loss_avg.update(total_loss.item(), B)
    return loss_avg.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune CLIP')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    print('{} FP16 training'.format('Not using' if args.fp32 else 'Using'))

    # DDP init
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{args.local_rank}')

    params = FinetuneParams()
    model, preproc, optimizer, start_epoch = build_model(params, args.weight)
    datamodule = build_datamodule(params, preproc)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # log
    start_datetime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    logs_dir = f"logs/{start_datetime}".format()
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    exp_name = f'finetune_clip-{SLURM_JOB_ID}'
    if args.local_rank == 0:
        os.makedirs(logs_dir, exist_ok=True)
        wandb.init(
            project='slot-attention-clevr6-language-video',
            name=exp_name,
            id=exp_name)

    torch.distributed.barrier()
    main()
