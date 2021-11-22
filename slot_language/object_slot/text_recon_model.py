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

    def __init__(self,
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
                 enc_resolution: Tuple[int, int] = (128, 128),
                 enc_channels: int = 3,
                 enc_pos_enc: bool = False,
                 slot_size: int = 64,
                 dec_kernel_size: int = 5,
                 dec_hidden_dims: Tuple[int, ...] = (64, 64, 64, 64, 64),
                 dec_resolution: Tuple[int, int] = (8, 8),
                 slot_mlp_size: int = 128,
                 use_entropy_loss: bool = False,
                 use_bg_sep_slot: bool = False):
        super().__init__(
            clip_model,
            use_clip_vision,
            use_clip_text,
            text2slot_model,
            resolution,
            num_slots,
            num_iterations,
            enc_resolution=enc_resolution,
            enc_channels=enc_channels,
            enc_pos_enc=enc_pos_enc,
            slot_size=slot_size,
            dec_kernel_size=dec_kernel_size,
            dec_hidden_dims=dec_hidden_dims,
            dec_resolution=dec_resolution,
            slot_mlp_size=slot_mlp_size,
            use_entropy_loss=use_entropy_loss,
            use_bg_sep_slot=use_bg_sep_slot)

        self.color_tokens = [1746, 2866, 1470, 7048, 1901, 5496, 736, 4481]
        self.shape_tokens = [10266, 11353, 22092,  6987] if \
            viewpoint_dataset else [11353, 22092]
        self.num_colors = len(self.color_tokens)
        self.num_shapes = len(self.shape_tokens)
        self.color_mlp = build_mlps(self.dec_hidden_dims[-1], cls_mlps,
                                    self.num_colors)
        self.shape_mlp = build_mlps(self.dec_hidden_dims[-1], cls_mlps,
                                    self.num_shapes)
        self.hard_visual_masking = hard_visual_masking
        assert recon_from in ['feats', 'slots', 'recons']
        self.recon_from = recon_from

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        if self.use_clip_vision:
            encoder_out = self.clip_model.encode_image(
                img, global_feats=False, downstream=True)  # BCDD
            encoder_out = encoder_out.type(self.dtype)
        else:
            encoder_out = self.encoder(img)
        img_feats = encoder_out  # Conv features without pos_enc
        # may not applying pos_enc because Encoder in CLIP already does so
        if self.enc_pos_enc:
            encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, C, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, C, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, C]
        return encoder_out, img_feats

    def forward(self, x):
        torch.cuda.empty_cache()

        slots, img_feats, obj_mask = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            slots, x['img'].shape)

        if not self.training:
            return recon_combined, recons, masks, slots

        pred_colors, pred_shapes, gt_colors, gt_shapes = self._reconstruct_text(
            img_feats, recons, masks, slots, x['text'], obj_mask)

        return recon_combined, recons, masks, slots, \
            pred_colors, pred_shapes, gt_colors, gt_shapes

    def encode(self, x):
        """Encode from img to slots."""
        img, text, padding = x['img'], x['text'], x['padding']
        encoder_out, img_feats = self._get_encoder_out(img)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]
        # `img_feats` has shape: [B, C, H, W]

        # slot initialization
        slot_mu, obj_mask = self._get_slot_embedding(text, padding)

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(encoder_out, slot_mu, fg_mask=obj_mask)
        return slots, img_feats, obj_mask

    def _reconstruct_text(self, img_feats, recon_imgs, slot_masks, slot_emb,
                          tokens, obj_mask):
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
        grouped_feats = grouped_feats[obj_mask]
        pred_colors = self.color_mlp(grouped_feats)
        pred_shapes = self.shape_mlp(grouped_feats)

        # construct labels
        obj_tokens = tokens[obj_mask]
        gt_colors = obj_tokens[:, 2]
        for i, color in enumerate(self.color_tokens):
            color_mask = (gt_colors == color)
            gt_colors[color_mask] = i
            # 'cyan' will be splited into two tokens
            # so the shape token is at [4]
            if color == 1470:
                obj_tokens[color_mask, 3] = obj_tokens[color_mask, 4]

        gt_shapes = obj_tokens[:, 3]
        for i, shape in enumerate(self.shape_tokens):
            gt_shapes[gt_shapes == shape] = i
        assert 0 <= gt_colors.min().item() <= gt_colors.max().item(
        ) < self.num_colors
        assert 0 <= gt_shapes.min().item() <= gt_shapes.max().item(
        ) < self.num_shapes

        return pred_colors, pred_shapes, gt_colors, gt_shapes

    def loss_function(self, input):
        if self.training:
            recon_combined, recons, masks, slots, pred_colors, pred_shapes, \
                gt_colors, gt_shapes = self.forward(input)
        else:
            recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input['img'])
        loss_dict = {
            'recon_loss': loss,
        }
        if self.training:
            color_cls_loss = F.cross_entropy(pred_colors, gt_colors)
            shape_cls_loss = F.cross_entropy(pred_shapes, gt_shapes)
            loss_dict['color_cls_loss'] = color_cls_loss
            loss_dict['shape_cls_loss'] = shape_cls_loss
        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy'] = entropy_loss
        return loss_dict


class ObjFeatPredSlotAttentionModel(ObjTwoClsSlotAttentionModel):

    def __init__(self,
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
                 enc_resolution: Tuple[int, int] = (128, 128),
                 enc_channels: int = 3,
                 enc_pos_enc: bool = False,
                 slot_size: int = 64,
                 dec_kernel_size: int = 5,
                 dec_hidden_dims: Tuple[int, ...] = (64, 64, 64, 64, 64),
                 dec_resolution: Tuple[int, int] = (8, 8),
                 slot_mlp_size: int = 128,
                 use_entropy_loss: bool = False,
                 use_bg_sep_slot: bool = False):
        ObjSlotAttentionModel.__init__(
            self,
            clip_model,
            use_clip_vision,
            use_clip_text,
            text2slot_model,
            resolution,
            num_slots,
            num_iterations,
            enc_resolution=enc_resolution,
            enc_channels=enc_channels,
            enc_pos_enc=enc_pos_enc,
            slot_size=slot_size,
            dec_kernel_size=dec_kernel_size,
            dec_hidden_dims=dec_hidden_dims,
            dec_resolution=dec_resolution,
            slot_mlp_size=slot_mlp_size,
            use_entropy_loss=use_entropy_loss,
            use_bg_sep_slot=use_bg_sep_slot)

        self.recon_mlp = build_mlps(self.slot_size, recon_mlps,
                                    self.text2slot_model.in_channels)
        self.hard_visual_masking = hard_visual_masking
        self.normalize_feats = normalize_feats
        assert recon_from in ['feats', 'slots', 'recons']
        self.recon_from = recon_from

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        if self.use_clip_vision:
            encoder_out = self.clip_model.encode_image(
                img, global_feats=False, downstream=True)  # BCDD
            encoder_out = encoder_out.type(self.dtype)
        else:
            encoder_out = self.encoder(img)
        img_feats = encoder_out  # Conv features without pos_enc
        # may not applying pos_enc because Encoder in CLIP already does so
        if self.enc_pos_enc:
            encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, C, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, C, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, C]
        return encoder_out, img_feats

    def _get_slot_embedding(self, tokens, paddings):
        """Encode text, generate slot embeddings.

        Args:
            tokens: [B, N, C]
            padding: [B, N]
        """
        if not self.use_clip_text:
            # not generating slots
            return None, None
        # we treat each obj as batch dim and get global text (for each phrase)
        obj_mask = (paddings == 1)
        obj_tokens = tokens[obj_mask]  # [K, C]
        text_features = self.clip_model.encode_text(
            obj_tokens, lin_proj=False, per_token_emb=False,
            return_mask=False)  # [K, C]
        text_features = text_features.type(self.dtype)
        slots = self.text2slot_model(text_features, obj_mask)
        return text_features, slots, obj_mask

    def forward(self, x):
        torch.cuda.empty_cache()

        slots, img_feats, text_feats, obj_mask = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            slots, x['img'].shape)

        if not self.training:
            return recon_combined, recons, masks, slots

        gt_text_feats, pred_text_feats = self._reconstruct_text(
            img_feats, text_feats, recons, masks, slots, obj_mask)

        return recon_combined, recons, masks, slots, \
            gt_text_feats, pred_text_feats

    def encode(self, x):
        """Encode from img to slots."""
        img, text, padding = x['img'], x['text'], x['padding']
        encoder_out, img_feats = self._get_encoder_out(img)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]
        # `img_feats` has shape: [B, C, H, W]

        # slot initialization
        # `text_feats` is of shape [K, C], corresponding to fg slots
        text_feats, slot_mu, obj_mask = self._get_slot_embedding(text, padding)

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(encoder_out, slot_mu, fg_mask=obj_mask)
        return slots, img_feats, text_feats, obj_mask

    def _reconstruct_text(self, img_feats, text_feats, recon_imgs, slot_masks,
                          slot_emb, obj_mask):
        """Reconstruct text from grouped visual features.

        Args:
            img_feats: [B, C, H, W]
            text_feats: [K, C]
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
        grouped_feats = grouped_feats[obj_mask]  # [K, C]
        pred_text_feats = self.recon_mlp(grouped_feats)

        if self.normalize_feats:
            text_feats = F.normalize(text_feats, p=2, dim=-1)
            pred_text_feats = F.normalize(pred_text_feats, p=2, dim=-1)

        return text_feats, pred_text_feats

    def loss_function(self, input):
        if self.training:
            recon_combined, recons, masks, slots, \
                gt_text_feats, pred_text_feats = self.forward(input)
        else:
            recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input['img'])
        loss_dict = {
            'recon_loss': loss,
        }
        if self.training:
            text_recon_loss = F.mse_loss(pred_text_feats, gt_text_feats)
            loss_dict['text_recon_loss'] = text_recon_loss
        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy'] = entropy_loss
        return loss_dict
