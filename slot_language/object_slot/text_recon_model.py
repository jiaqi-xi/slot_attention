from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from obj_utils import build_mlps
from obj_model import ObjSlotAttentionModel


class ObjTwoClsSlotAttentionModel(ObjSlotAttentionModel):
    """Reconstruct text from grouped features.

    The reconstructed task is simplified to two classification:
        - color: ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
            their token_id are [1746, 2866, 1470, 7048, 1901, 5496,  736, 4481]
        - shape: ['cube', 'cylinder']
            their token_id are [11353, 22092]
        *Note*: for `viewpoint_dataset`, shapes are ['cone', 'cube', 'cylinder', 'sphere']
            their token_id are [10266, 11353, 22092,  6987]
    """

    def __init__(
        self,
        clip_model: CLIP,
        use_clip_vision: bool,
        use_clip_text: bool,
        text2slot_model: nn.Module,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations: int,
        viewpoint_dataset: bool = False,
        cls_mlps: Tuple[int, ...] = (),
        hard_visual_masking: bool = False,
        recon_from: str = 'feats',
        slot_size: int = 64,
        slot_mlp_size: int = 128,
        out_features: int = 64,
        kernel_size: int = 5,
        use_unet: bool = False,
        enc_channels: Tuple[int, ...] = ...,
        dec_channels: Tuple[int, ...] = ...,
        dec_resolution: Tuple[int, int] = ...,
        use_bg_sep_slot: bool = False,
        enc_resolution: Tuple[int, int] = ...,
        visual_feats_channels: int = 512,
        use_entropy_loss: bool = False,
    ):
        super().__init__(
            clip_model,
            use_clip_vision,
            use_clip_text,
            text2slot_model,
            resolution,
            num_slots,
            num_iterations,
            slot_size=slot_size,
            slot_mlp_size=slot_mlp_size,
            out_features=out_features,
            kernel_size=kernel_size,
            use_unet=use_unet,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            dec_resolution=dec_resolution,
            use_bg_sep_slot=use_bg_sep_slot,
            enc_resolution=enc_resolution,
            visual_feats_channels=visual_feats_channels,
            use_entropy_loss=use_entropy_loss,
        )

        self.color_tokens = torch.tensor(
            [1746, 2866, 1470, 7048, 1901, 5496, 736, 4481])
        self.shape_tokens = torch.tensor([10266, 11353, 22092,  6987]) if \
            viewpoint_dataset else torch.tensor([11353, 22092])
        self.num_colors = len(self.color_tokens)
        self.num_shapes = len(self.shape_tokens)
        self.color_mlp = build_mlps(self.dec_hidden_dims[-1], cls_mlps,
                                    self.num_colors + 1)
        self.shape_mlp = build_mlps(self.dec_hidden_dims[-1], cls_mlps,
                                    self.num_shapes + 1)
        self.hard_visual_masking = hard_visual_masking
        assert recon_from in ['feats', 'slots', 'recons']
        self.recon_from = recon_from

    def forward(self, x):
        torch.cuda.empty_cache()

        slots, img_feats, text_feats = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            slots, x['img'].shape)

        if not self.training:
            return recon_combined, recons, masks, slots

        pred_colors, pred_shapes, gt_colors, gt_shapes = self._reconstruct_text(
            img_feats, recons, masks, slots, x['text'])

        return recon_combined, recons, masks, slots, img_feats, text_feats, \
            pred_colors, pred_shapes, gt_colors, gt_shapes

    def _reconstruct_text(self, img_feats, recon_imgs, slot_masks, slot_emb,
                          tokens):
        """Reconstruct text from grouped visual features.

        Args:
            img_feats: [B, C, H, W]
            recon_imgs: [B, num_slots, C, H, W]
            slots_masks: [B, num_slots, 1, H, W]
            slot_emb: [B, num_slots, C]
            tokens: [B, num_slots, 77], color at [2], shape at [3]
            obj_mask: [B, num_slots]
        """
        B, C, H, W = img_feats.shape
        if self.recon_from == 'feats':
            slot_masks = slot_masks[:, :, 0]  # [B, num_slots, H, W]
            if self.hard_visual_masking:
                slot_masks = (slot_masks == slot_masks.max(1, keepdim=True)[0])
                slot_masks = slot_masks.as_type(img_feats)
            grouped_feats = img_feats[:, None] * slot_masks[:, :, None]
            grouped_feats = grouped_feats.sum(dim=[-1, -2]) / \
                (slot_masks.sum(dim=[-1, -2])[:, :, None] + 1e-6)
        elif self.recon_from == 'recons':
            recon_imgs = recon_imgs * slot_masks  # [B, num_slots, C, H, W]
            img_feats = self.encoder(recon_imgs.flatten(0, 1)).view(
                B, self.num_slots, C, H, W)
            grouped_feats = img_feats.mean(dim=[-1, -2])
        else:
            grouped_feats = slot_emb

        assert grouped_feats.shape == torch.Size([B, self.num_slots, C])
        obj_mask = torch.ones(
            (tokens.shape[0], tokens.shape[1])).type_as(tokens).bool()
        grouped_feats = grouped_feats[obj_mask]
        pred_colors = self.color_mlp(grouped_feats)
        pred_shapes = self.shape_mlp(grouped_feats)

        # construct labels
        obj_tokens = tokens[obj_mask].detach().clone()
        gt_colors = obj_tokens[:, 2].detach().clone()
        color_tokens = obj_tokens[:, 2].detach().clone()
        for i, color in enumerate(self.color_tokens):
            color_mask = (color_tokens == color)
            gt_colors[color_mask] = i
            # 'cyan' will be splited into two tokens
            # so the shape token is at [4]
            if color == 1470:
                obj_tokens[color_mask, 3] = obj_tokens[color_mask, 4]
        # for background texts
        gt_colors[~torch.
                  isin(color_tokens, self.color_tokens.type_as(color_tokens)
                       )] = self.num_colors

        gt_shapes = obj_tokens[:, 3].detach().clone()
        shape_tokens = obj_tokens[:, 3].detach().clone()
        for i, shape in enumerate(self.shape_tokens):
            gt_shapes[shape_tokens == shape] = i
        gt_shapes[~torch.
                  isin(shape_tokens, self.shape_tokens.type_as(shape_tokens)
                       )] = self.num_shapes

        assert 0 <= gt_colors.min().item() <= gt_colors.max().item(
        ) <= self.num_colors
        assert 0 <= gt_shapes.min().item() <= gt_shapes.max().item(
        ) <= self.num_shapes

        return pred_colors, pred_shapes, gt_colors, gt_shapes

    def loss_function(self, input):
        if not self.training:
            return self.eval_loss_function(input)

        recon_combined, recons, masks, slots, img_feats, text_feats, \
            pred_colors, pred_shapes, gt_colors, gt_shapes = self.forward(input)
        loss_dict = self.calc_train_loss(input['img'], recon_combined, recons,
                                         masks, slots, img_feats, text_feats)
        color_cls_loss = F.cross_entropy(pred_colors, gt_colors)
        shape_cls_loss = F.cross_entropy(pred_shapes, gt_shapes)
        loss_dict['color_cls_loss'] = color_cls_loss
        loss_dict['shape_cls_loss'] = shape_cls_loss
        return loss_dict


class ObjFeatPredSlotAttentionModel(ObjTwoClsSlotAttentionModel):

    def __init__(
        self,
        clip_model: CLIP,
        use_clip_vision: bool,
        use_clip_text: bool,
        text2slot_model: nn.Module,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations: int,
        recon_mlps: Tuple[int, ...] = (),
        hard_visual_masking: bool = False,
        normalize_feats: bool = False,
        recon_from: str = 'feats',
        slot_size: int = 64,
        slot_mlp_size: int = 128,
        out_features: int = 64,
        kernel_size: int = 5,
        use_unet: bool = False,
        enc_channels: Tuple[int, ...] = ...,
        dec_channels: Tuple[int, ...] = ...,
        dec_resolution: Tuple[int, int] = ...,
        use_bg_sep_slot: bool = False,
        enc_resolution: Tuple[int, int] = ...,
        visual_feats_channels: int = 512,
        use_entropy_loss: bool = False,
    ):
        ObjSlotAttentionModel.__init__(
            self,
            clip_model,
            use_clip_vision,
            use_clip_text,
            text2slot_model,
            resolution,
            num_slots,
            num_iterations,
            slot_size=slot_size,
            slot_mlp_size=slot_mlp_size,
            out_features=out_features,
            kernel_size=kernel_size,
            use_unet=use_unet,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            dec_resolution=dec_resolution,
            use_bg_sep_slot=use_bg_sep_slot,
            enc_resolution=enc_resolution,
            visual_feats_channels=visual_feats_channels,
            use_entropy_loss=use_entropy_loss,
        )

        self.recon_mlp = build_mlps(self.slot_size, recon_mlps,
                                    self.text2slot_model.in_channels)
        self.hard_visual_masking = hard_visual_masking
        self.normalize_feats = normalize_feats
        assert recon_from in ['feats', 'slots', 'recons']
        self.recon_from = recon_from

    def forward(self, x):
        torch.cuda.empty_cache()

        slots, img_feats, text_feats = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            slots, x['img'].shape)

        if not self.training:
            return recon_combined, recons, masks, slots

        gt_text_feats, pred_text_feats = self._reconstruct_text(
            img_feats, text_feats, recons, masks, slots)

        return recon_combined, recons, masks, slots, img_feats, text_feats, \
            gt_text_feats, pred_text_feats

    def _reconstruct_text(self, img_feats, text_feats, recon_imgs, slot_masks,
                          slot_emb):
        """Reconstruct text from grouped visual features.

        Args:
            img_feats: [B, C, H, W]
            text_feats: [B, num_slots, C]
            recon_imgs: [B, num_slots, C, H, W]
            slots_masks: [B, num_slots, 1, H, W]
            slot_emb: [B, num_slots, C]
            obj_mask: [B, num_slots]
        """
        B, C, H, W = img_feats.shape
        if self.recon_from == 'feats':
            slot_masks = slot_masks[:, :, 0]  # [B, num_slots, H, W]
            if self.hard_visual_masking:
                slot_masks = (slot_masks == slot_masks.max(1, keepdim=True)[0])
                slot_masks = slot_masks.as_type(img_feats)
            grouped_feats = img_feats[:, None] * slot_masks[:, :, None]
            grouped_feats = grouped_feats.sum(dim=[-1, -2]) / \
                (slot_masks.sum(dim=[-1, -2])[:, :, None] + 1e-6)
        elif self.recon_from == 'recons':
            recon_imgs = recon_imgs * slot_masks  # [B, num_slots, C, H, W]
            img_feats = self.encoder(recon_imgs.flatten(0, 1)).view(
                B, self.num_slots, C, H, W)
            grouped_feats = img_feats.mean(dim=[-1, -2])
        else:
            grouped_feats = slot_emb

        assert grouped_feats.shape == torch.Size([B, self.num_slots, C])
        obj_mask = torch.ones(
            (slot_emb.shape[0], slot_emb.shape[1])).type_as(slot_emb).bool()
        grouped_feats = grouped_feats[obj_mask]  # [K, C]
        pred_text_feats = self.recon_mlp(grouped_feats)
        text_feats = text_feats[obj_mask]  # [K, C]

        if self.normalize_feats:
            text_feats = F.normalize(text_feats, p=2, dim=-1)
            pred_text_feats = F.normalize(pred_text_feats, p=2, dim=-1)

        return text_feats, pred_text_feats

    def loss_function(self, input):
        if not self.training:
            return self.eval_loss_function(input)

        recon_combined, recons, masks, slots, img_feats, text_feats, \
            gt_text_feats, pred_text_feats = self.forward(input)
        loss_dict = self.calc_train_loss(input['img'], recon_combined, recons,
                                         masks, slots, img_feats, text_feats)
        text_recon_loss = F.mse_loss(pred_text_feats, gt_text_feats)
        loss_dict['text_recon_loss'] = text_recon_loss
        return loss_dict
