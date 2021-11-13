import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from obj_model import ObjSlotAttentionModel


class ObjPosSlotAttentionModel(ObjSlotAttentionModel):

    def __init__(
            self,
            clip_model: CLIP,
            use_clip_vision: bool,
            use_clip_text: bool,
            text2slot_model: nn.Module,
            resolution: Tuple[int, int],
            num_slots: int,
            num_iterations: int,
            enc_resolution: Tuple[int, int] = (128, 128),
            num_pos_slot: int = 5,  # number of pos enc for each slot
            enc_channels: int = 3,
            enc_pos_enc: bool = False,
            slot_size: int = 64,
            dec_kernel_size: int = 5,
            dec_hidden_dims: Tuple[int, ...] = (64, 64, 64, 64, 64),
            dec_resolution: Tuple[int, int] = (8, 8),
            slot_mlp_size: int = 128,
            use_word_set: bool = False,
            use_padding_mask: bool = False,
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
            use_word_set=use_word_set,
            use_padding_mask=use_padding_mask,
            use_entropy_loss=use_entropy_loss,
            use_bg_sep_slot=use_bg_sep_slot)

        self.slot_attention.num_slots *= num_pos_slot
        self.num_pos_slot = num_pos_slot
        self.pos_slot_emb = nn.Embedding(num_slots * num_pos_slot, slot_size)
        nn.init.xavier_uniform_(
            self.pos_slot_emb.weight, gain=nn.init.calculate_gain("linear"))
        self.slot_proj = nn.Linear(2 * slot_size, slot_size, bias=True)

    def encode(self, x):
        """Encode from img to slots."""
        img, text, padding = x['img'], x['text'], x['padding']
        bs = img.shape[0]
        encoder_out = self._get_encoder_out(img)  # transformed vision feature
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        # slot initialization, text_slots of shape [B, num_slots, slot_size]
        text_slots, obj_mask = self._get_slot_embedding(text, padding)

        # concat text_slots with pos_slot_emb
        text_slots = text_slots.unsqueeze(2).\
            repeat(1, 1, self.num_pos_slot, 1).flatten(1, 2)
        pos_slots = self.pos_slot_emb.weight.unsqueeze(0).repeat(bs, 1, 1)
        slots = torch.cat([text_slots, pos_slots], dim=-1)
        slots = self.slot_proj(slots)
        obj_mask = obj_mask.unsqueeze(2).\
            repeat(1, 1, self.num_pos_slot).flatten(1, 2)

        # (batch_size, self.num_slots * self.num_pos_slot, self.slot_size)
        slots = self.slot_attention(encoder_out, slots, fg_mask=obj_mask)
        return slots

    def decode(self, slots, img_shape):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [B, num_slots, slot_size].
        bs, num_slots, slot_size = slots.shape
        assert num_slots == self.num_slots * self.num_pos_slot
        bs, C, H, W = img_shape

        # spatial broadcast
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, C+1, H, W].

        out = out.view(bs, self.num_slots, self.num_pos_slot, C + 1, H, W)
        recons = out[:, :, :, :C, :, :]
        masks = out[:, :, :, -1:, :, :]

        # sum over each pos_emb
        pos_masks = F.softmax(masks, dim=2)
        recons = (recons * pos_masks).sum(2)  # [B, num_slots, C, H, W]
        # masks = (masks * pos_masks).sum(2)  # [B, num_slots, 1, H, W]
        # masks = F.softmax(masks, dim=1)
        all_slots_masks = F.softmax(
            masks.flatten(1, 2), dim=1).view(masks.shape)
        masks = all_slots_masks.sum(2)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, all_slots_masks
