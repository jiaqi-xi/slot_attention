from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from obj_model import ObjSlotAttentionModel


class ObjRecurSlotAttentionModel(ObjSlotAttentionModel):

    def __init__(self,
                 clip_model: CLIP,
                 use_clip_vision: bool,
                 use_clip_text: bool,
                 text2slot_model: nn.Module,
                 resolution: Tuple[int, int],
                 num_slots: int,
                 num_iterations: int,
                 enc_resolution: Tuple[int, int] = (7, 7),
                 enc_channels: int = 3,
                 enc_pos_enc: bool = False,
                 slot_size: int = 64,
                 dec_kernel_size: int = 5,
                 dec_hidden_dims: Tuple[int, ...] = (64, 64, 64, 64, 64),
                 dec_resolution: Tuple[int, int] = (7, 7),
                 slot_mlp_size: int = 128,
                 use_entropy_loss: bool = False,
                 use_bg_sep_slot: bool = False,
                 slot_emb_lstm: nn.LSTM = None):
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
        self.slot_emb_lstm = slot_emb_lstm

    def forward(self, x):
        torch.cuda.empty_cache()

        slots = self.encode(x)

        N = slots.shape[0]
        C, H, W = x['img'].shape[-3:]
        recon_combined, recons, masks, slots = self.decode(slots, (N, C, H, W))
        return recon_combined, recons, masks, slots

    def encode(self, x):
        """Encode from img to slots."""
        img, text, padding = x['img'], x['text'], x['padding']
        bs, sample_num, C, H, W = img.shape
        encoder_out = self._get_encoder_out(img.view(-1, C, H, W)).view(
            bs, sample_num, H * W, -1)
        # `encoder_out` has shape: [batch_size, num, height*width, filter_size]

        # slot initialization
        slot_mu, obj_mask = self._get_slot_embedding(text, padding)

        # apply SlotAttention iteratively
        # `slot_mu` has shape: [batch_size, num_slots, slot_size]
        slots, slot_prev = [], slot_mu
        if self.slot_emb_lstm is not None:
            (h, c) = self._init_lstm_hidden(slot_mu)
        for clip_idx in range(sample_num):
            one_slot = self.slot_attention(
                encoder_out[:, clip_idx], slot_prev, fg_mask=obj_mask)
            slots.append(one_slot)
            slot_prev = one_slot
            # stop the grad of slot_prev
            slot_prev = slot_prev.detach().clone()
            if self.slot_emb_lstm is not None:
                slot_prev, (h, c) = self._lstm_update(slot_prev, h, c)
        slots = torch.stack(slots, dim=1)  # [bs, num, num_slots, slot_size]
        return slots.flatten(0, 1)  # [N, num_slots, slot_size]

    def _init_lstm_hidden(self, input):
        """Zero init hidden (h) and cell (c) state for LSTM."""
        assert self.slot_emb_lstm is not None
        num_directions = 2 if self.slot_emb_lstm.bidirectional else 1
        real_hidden_size = self.slot_emb_lstm.proj_size if \
            self.slot_emb_lstm.proj_size > 0 else self.slot_emb_lstm.hidden_size
        batch_size = input.shape[0]
        h_zeros = torch.zeros(
            self.slot_emb_lstm.num_layers * num_directions,
            self.num_slots * batch_size,
            real_hidden_size,
            dtype=input.dtype,
            device=input.device).detach()
        c_zeros = torch.zeros(
            self.slot_emb_lstm.num_layers * num_directions,
            self.num_slots * batch_size,
            self.slot_emb_lstm.hidden_size,
            dtype=input.dtype,
            device=input.device).detach()
        return (h_zeros, c_zeros)

    def _lstm_update(self, slot_emb, h, c):
        """One step forward of LSTM."""
        # slot_emb has shape: [batch_size, num_slots, slot_size]
        batch_size = slot_emb.shape[0]
        slot_emb = slot_emb.view(batch_size * self.num_slots, self.slot_size)
        dim = 1 if self.slot_emb_lstm.batch_first else 0
        slot_emb = slot_emb.unsqueeze(dim)  # input length L = 1
        update_slot_emb, (h, c) = self.slot_emb_lstm(slot_emb, (h, c))
        update_slot_emb = update_slot_emb.squeeze(dim).view(
            batch_size, self.num_slots, self.slot_size)
        return update_slot_emb, (h, c)

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input['img'].flatten(0, 1))
        loss_dict = {
            "recon_loss": loss,
        }
        # masks: [N, num_slots, 1, H, W], apply entropy loss
        if self.use_entropy_loss:
            masks = masks[:, :, 0]  # [N, num_slots, H, W]
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy'] = entropy_loss
        return loss_dict
