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

from obj_data import ObjCLEVRVisionLanguageCLIPDataset
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
    return seg_masks, [sim_map[i][obj_mask[i]] for i in range(B)]


def main(params: SegParams):
    # clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocesser = clip.load(params.clip_arch, device=device)

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
    test(clip_model, val_loader, val_dataset, params)


def test(clip_model, dataloader, dataset, params: SegParams):
    dataloader = iter(dataloader)
    batch_data = next(dataloader)
    batch_data = {k: v[:params.num_test] for k, v in batch_data.items()}
    with torch.no_grad():
        # get `seg_mask` of shape [B, n, n], n is number of grids
        seg_masks, probs_masks = batch_seg(clip_model, batch_data)
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
        pil_mask = Image.fromarray(
            cv2.resize(
                colored_seg_mask,
                pil_img.size,
                interpolation=cv2.INTER_NEAREST))
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
        ax2.legend(bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(vis_path, f'{i}.png'))
        plt.close(fig)
        print(raw_texts[i], f'segmenting {seg_masks[i].max() + 1} classes')
        # plot prob_mask for each text, [K, n, n]
        probs_mask = probs_masks[i]
        for j in range(probs_mask.shape[0]):
            probs_mask[j] += probs_mask[j].min()
            probs_mask[j] /= probs_mask[j].max()  # to [0, 1]
            probs_mask[j] = (probs_mask[j] * 255.).astype(np.uint8)
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
        fig.savefig(os.path.join(vis_path, f'{i}_probs.png'))
        plt.close(fig)


if __name__ == '__main__':
    vis_path = './vis/'
    os.makedirs(vis_path, exist_ok=True)
    params = SegParams()
    main(params)
