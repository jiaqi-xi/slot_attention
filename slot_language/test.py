import os
import importlib
import argparse
import numpy as np

import torch
from torchvision import utils as vutils

import clip
from text_model import MLPText2Slot, TransformerText2Slot
from detr_module import DETRText2Slot
from data import CLEVRVisionLanguageCLIPDataModule
from method import SlotAttentionVideoLanguageMethod as SlotAttentionMethod
from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import to_rgb_from_tensor, save_video, simple_rescale


def main(params=None):
    if params is None:
        params = SlotAttentionParams()

    clip_model, clip_transforms = clip.load(params.clip_arch)
    if not params.use_clip_vision:
        from torchvision.transforms import Compose, Resize, ToTensor, \
            Normalize, Lambda
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        normalize = Lambda(
            simple_rescale) if params.simple_normalize else Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711))
        clip_transforms = Compose([
            Resize(params.resolution, interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            normalize,
        ])

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

    model = SlotAttentionModel(
        clip_model=clip_model,
        use_clip_vision=params.use_clip_vision,
        use_clip_text=params.use_text2slot,
        text2slot_model=text2slot_model,
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        enc_resolution=params.enc_resolution,
        enc_channels=params.clip_vision_channel,
        enc_pos_enc=params.enc_pos_enc,
        slot_size=params.slot_size,
        dec_kernel_size=params.dec_kernel_size,
        dec_hidden_dims=params.dec_channels,
        dec_resolution=params.dec_resolution,
        slot_mlp_size=params.slot_mlp_size,
        use_word_set=params.use_text2slot
        and params.text2slot_arch in ['Transformer', 'DETR'],
        use_padding_mask=params.use_text2slot
        and params.text2slot_arch in ['Transformer', 'DETR']
        and params.text2slot_padding_mask,
    )

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
        separater=params.separater,
    )

    model = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params)
    model.load_state_dict(torch.load(args.weight)['state_dict'], strict=True)
    model = model.cuda().eval()

    save_folder = os.path.join(os.path.dirname(args.weight), 'vis')
    os.makedirs(save_folder, exist_ok=True)

    # get image from train and val dataset
    with torch.no_grad():
        train_res = inference(
            model, clevr_datamodule.train_dataset, num=args.test_num)
        val_res = inference(
            model, clevr_datamodule.val_dataset, num=args.test_num)
    save_video(train_res, os.path.join(save_folder, 'train.mp4'), fps=2)
    save_video(val_res, os.path.join(save_folder, 'val.mp4'), fps=2)


def inference(model, dataset, num=3):
    dataset.is_video = True
    num_data = dataset.num_videos
    data_idx = np.random.choice(num_data, num, replace=False)
    results = []
    all_texts = []
    for idx in data_idx:
        batch = dataset.__getitem__(idx)  # dict with key video, text, raw_text
        video, text, raw_text = \
            batch['video'], batch['text'], batch['raw_text']
        all_texts.append(raw_text)
        batch = dict(img=video.float().cuda(), text=text.float().cuda())
        recon_combined, recons, masks, slots = model(batch)
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch['img'].unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            ))  # [B (temporal dim), num_slots+2, 3, H, W]

        T, num_slots, C, H, W = recons.shape
        video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                normalize=False,
                nrow=out.shape[1],
            ) for i in range(T)
        ])  # [T, 3, H, (num_slots+2)*W]
        results.append(video.numpy())

    # concat results vertically
    results = np.concatenate(results, axis=2)  # [T, 3, B*H, (num_slots+2)*W]
    results = np.ascontiguousarray(results.transpose((0, 2, 3, 1)))
    return results


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    parser = argparse.ArgumentParser(description='Test Slot Attention')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--test-num', type=int, default=5)
    # TODO: I didn't find improvement using num-iter=5 as stated in the paper
    parser.add_argument('--num-iter', type=int, default=3)
    args = parser.parse_args()
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    params = importlib.import_module(args.params)
    params = params.SlotAttentionParams()
    params.num_iterations = args.num_iter
    main(params)
