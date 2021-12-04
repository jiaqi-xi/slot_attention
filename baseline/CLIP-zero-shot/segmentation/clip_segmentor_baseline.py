import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from obj_data import ObjCLEVRVisionLanguageCLIPDataset, build_data_transforms
from seg_params import SegParams


def batch_seg(clip_model, batch_data):
    # img: [B, C, H, W]
    # tokens: [B, num_obj, 77]
    batch_data = {k: v.cuda() for k, v in batch_data.items()}
    img, tokens = batch_data['img'], batch_data['tokens']
    B = img.shape[0]
    # encode image features
    img_feats = clip_model.encode_image(
        img, global_feats=False, downstream=False)  # [B, n^2, C]
    num_grids = int(img_feats.shape[1]**0.5)
    img_feats = F.normalize(
        img_feats, p=2, dim=-1).permute(0, 2, 1).contiguous()  # [B, C, n^2]
    # encode text features
    text_feats = clip_model.encode_text(
        tokens.view(-1, tokens.shape[-1]),
        lin_proj=True,
        per_token_emb=False,
        return_mask=False)  # [B*num_obj, C]
    text_feats = F.normalize(text_feats, p=2, dim=-1).\
        view(B, -1, text_feats.shape[-1])  # [B, num_obj, C]
    assert img_feats.shape[1] == text_feats.shape[-1]
    # compute similarity map
    sim_map = (img_feats[:, None] * text_feats[:, :, :, None]).sum(2).reshape(
        B, -1, num_grids, num_grids)  # [B, num_obj, n, n]
    # convert to segmentation mask via argmax
    obj_mask = batch_data['obj_mask']  # [B, num_obj]
    seg_masks = torch.stack(
        [sim_map[i][obj_mask[i]].argmax(0) for i in range(B)],
        dim=0)  # [B, n, n]
    return seg_masks


def main(params: SegParams):
    # clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(params.clip_arch, device=device)

    # build dataloader
    transforms = build_data_transforms(params)
    val_dataset = ObjCLEVRVisionLanguageCLIPDataset(
        params.data_root,
        None,
        transforms,
        max_n_objects=params.max_n_objects,
        split='val',
        clip_len=34)
    val_loader = DataLoader(
        val_dataset,
        params.batch_size,
        False,
        num_workers=params.num_workers,
        pin_memory=True)
    test(clip_model, val_loader, params)


def test(clip_model, dataloader, params: SegParams):
    dataloader = iter(dataloader)
    batch_data = next(dataloader)
    batch_data = {k: v[:params.num_test] for k, v in batch_data.items()}
    with torch.no_grad():
        # get `seg_mask` of shape [B, n, n], n is number of grids
        seg_masks = batch_seg(clip_model, batch_data).detach().cpu().numpy()
    imgs = batch_data['ori_img'].detach().cpu().numpy().astype(np.uint8)
    data_idx = batch_data['data_idx'].detach().cpu().numpy()
    raw_texts = [
        dataloader.dataset._generate_text(idx, padding=False)
        for idx in data_idx
    ]
    palette = np.array(params.PALETTE).astype(np.uint8)
    colored_seg_masks = palette[seg_masks]  # [B, n, n, 3]
    # visualize and save
    for i in range(params.num_test):
        img = Image.fromarray(imgs[i])
        colored_seg_mask = colored_seg_masks[i]  # [n, n, 3]
        mask = Image.fromarray(
            cv2.resize(
                colored_seg_mask, img.size, interpolation=cv2.INTER_NEAREST))
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img)
        ax2.imshow(mask)
        fig.savefig(os.path.join(vis_path, f'{i}.png'))
        plt.close(fig)
        print(raw_texts[i], f'segmenting {seg_masks[i].max() + 1} classes')


if __name__ == '__main__':
    vis_path = './vis/'
    os.makedirs(vis_path, exist_ok=True)
    params = SegParams()
    main(params)
