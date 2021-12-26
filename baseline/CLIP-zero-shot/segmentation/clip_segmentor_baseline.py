import os
import cv2
import random
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clip_model import CLIPVisionEncoder, CLIPTextEncoder
from obj_data import ObjCLEVRVisionLanguageCLIPDataset
from seg_params import SegParams


def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_seg(clip_vision, clip_text, batch_data):
    # img: [B, C, H, W]
    # tokens: [B, num_obj, 77]
    batch_data = {k: v.cuda() for k, v in batch_data.items()}
    img, tokens = batch_data['img'], batch_data['tokens']
    B = img.shape[0]
    # encode image features, [B, C, n1, n2]
    img_feats = clip_vision(img, lin_proj=True, res_no_pool=True)
    img_feats = F.normalize(img_feats, p=2, dim=1)
    n1, n2 = img_feats.shape[-2:]
    # encode text features, [B*num_obj, C]
    text_feats = clip_text(tokens.view(-1, tokens.shape[-1]), lin_proj=True)
    text_feats = F.normalize(text_feats, p=2, dim=-1).\
        view(B, -1, text_feats.shape[-1])  # [B, num_obj, C]
    assert img_feats.shape[1] == text_feats.shape[-1]
    # compute similarity map
    sim_map = (img_feats[:, None] * text_feats[:, :, :, None, None]).sum(2).\
        reshape(B, -1, n1, n2) * args.tau  # [B, num_obj, n1, n2]
    # convert to segmentation mask via argmax
    obj_mask = batch_data['obj_mask']  # [B, num_obj]
    seg_masks = torch.stack(
        [sim_map[i][obj_mask[i]].argmax(0) for i in range(B)],
        dim=0)  # [B, n1, n2]
    per_obj_masks = [
        torch.softmax(sim_map[i][obj_mask[i]], dim=0) for i in range(B)
    ]  # [B, num_fg_obj, n1, n2]
    return seg_masks, per_obj_masks


def main(params: SegParams):
    # clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocesser = clip.load(params.clip_arch, device=device)
    if 'vit' not in params.clip_arch.lower():
        # we don't need resizing/center crop in ResNet encoder
        preprocesser.transforms = preprocesser.transforms[2:]
    # load finetuned CLIP weight
    if args.weight:
        ckp = torch.load(args.weight, map_location='cpu')
        clip_model.load_state_dict(ckp['model_state_dict'])

    clip_vision = CLIPVisionEncoder(clip_model)
    clip_text = CLIPTextEncoder(clip_model)
    args.tau = clip_model.logit_scale.exp().item() if args.tau else 1.0

    # build dataloader
    val_dataset = ObjCLEVRVisionLanguageCLIPDataset(
        params.data_root,
        None,
        preprocesser,
        max_n_objects=params.max_n_objects,
        split='val',
        clip_len=34,
        prompt=params.prompt,
        pad_text=params.pad_text)
    val_loader = DataLoader(
        val_dataset,
        params.batch_size,
        False,
        num_workers=params.num_workers,
        pin_memory=True)
    test(clip_vision, clip_text, val_loader, val_dataset, params)


def test(clip_vision, clip_text, dataloader, dataset, params: SegParams):
    set_seed(0)
    dataloader = iter(dataloader)
    batch_data = next(dataloader)
    batch_data = {k: v[:params.num_test] for k, v in batch_data.items()}
    with torch.no_grad():
        # get `seg_mask` of shape [B, n, n], n is number of grids
        seg_masks, probs_masks = batch_seg(clip_vision, clip_text, batch_data)
        seg_masks = seg_masks.detach().cpu().numpy()
        probs_masks = [mask.detach().cpu().numpy() for mask in probs_masks]
    imgs = batch_data['ori_img'].detach().cpu().numpy().astype(np.float32)
    data_idx = batch_data['data_idx'].detach().cpu().numpy()
    raw_texts = [
        dataset._generate_text(idx, padding=False) for idx in data_idx
    ]
    palette = np.array(params.PALETTE).astype(np.uint8)
    colored_seg_masks = palette[seg_masks]  # [B, n, n, 3]
    # visualize and save
    for i in range(params.num_test):
        img = imgs[i]
        pil_img = Image.fromarray(img.astype(np.uint8))
        colored_seg_mask = colored_seg_masks[i]  # [n, n, 3]
        colored_seg_mask = cv2.resize(
            colored_seg_mask, pil_img.size, interpolation=cv2.INTER_NEAREST)
        blend_mask = colored_seg_mask.astype(np.float32) * 0.6 + img * 0.4
        pil_mask = Image.fromarray(blend_mask.astype(np.uint8))
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(pil_img)
        ax2.imshow(pil_mask)
        # in order to show the color for each text
        for lbl in np.unique(seg_masks[i]):
            ax2.plot([], [],
                     color=(palette[lbl].astype(np.float32) / 255.).tolist(),
                     label=raw_texts[i][lbl])
        ax2.legend(bbox_to_anchor=(1, 0.2))
        fig.savefig(os.path.join(vis_path, f'{i}.png'))
        plt.close(fig)
        print(raw_texts[i], f'segmenting {seg_masks[i].max() + 1} classes')
        # plot prob_mask for each text, [K, n, n]
        probs_mask = probs_masks[i]
        # TODO: because we already do softmax?
        # for j in range(probs_mask.shape[0]):
        #     probs_mask[j] += probs_mask[j].min()
        #     probs_mask[j] /= probs_mask[j].max()  # to [0, 1]
        assert probs_mask.min() >= 0. and probs_mask.max() <= 1.
        probs_mask = (probs_mask * 255.).astype(np.uint8)
        probs_mask = np.ascontiguousarray(probs_mask.transpose(1, 2, 0))
        probs_mask = cv2.resize(
            probs_mask.astype(np.uint8),
            pil_img.size,
            interpolation=cv2.INTER_NEAREST).astype(np.float32)
        probs_mask = np.stack([probs_mask] * 3, axis=2)  # [H, W, 3, K]
        fig = plt.figure(figsize=(18, 8))
        for j in range(probs_mask.shape[-1]):
            mask = probs_mask[..., j]
            ax = fig.add_subplot(241 + j)
            ax.set_title(raw_texts[i][j])
            blend_img = mask * 0.6 + img * 0.4
            ax.imshow(Image.fromarray(blend_img.astype(np.uint8)))
        # put the blended image and whole seg_mask
        ax = fig.add_subplot(248)
        ax.imshow(pil_mask)
        for lbl in np.unique(seg_masks[i]):
            ax.plot([], [],
                    color=(palette[lbl].astype(np.float32) / 255.).tolist(),
                    label=raw_texts[i][lbl])
        ax.legend(bbox_to_anchor=(1, 0.2), fontsize='small')
        fig.savefig(os.path.join(vis_path, f'{i}_probs.png'))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP for zero-shot seg')
    parser.add_argument('--params', type=str, default='seg_params')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--tau', action='store_true')
    args = parser.parse_args()
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    params = importlib.import_module(args.params)
    params = params.SegParams()
    vis_path = f'./vis/{args.params}'
    os.makedirs(vis_path, exist_ok=True)
    main(params)
