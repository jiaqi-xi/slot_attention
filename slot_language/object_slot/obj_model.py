import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from clip import CLIP

sys.path.append('../')

from model import SlotAttention, BgSepSlotAttention, SlotAttentionModel


class ObjSlotAttention(SlotAttention):
    """A wrapper for SlotAttention to make forward interface consistent."""

    def forward(self, inputs, slots_mu, slots_log_sigma=None, fg_mask=None):
        return super().forward(inputs, slots_mu, slots_log_sigma)


class ObjBgSepSlotAttention(BgSepSlotAttention):
    """Slot attention module that iteratively performs cross-attention.

    The BgSep one processes fg slots and bg slots seperately.
    TODO: the different of `Obj` version is that, here we may have different
        number of background slots among batch data. Fortunately, all the
        operations here are performed along the last dim (slot_size),
        so we can safely view slots to [N, slot_size] tensor and forward pass.
    """

    def forward(self, inputs, slots_mu, slots_log_sigma=None, fg_mask=None):
        """Forward function.

        Args:
            inputs: [B, N, C], flattened per-pixel features
            slots_mu: if [B, num_slots, C], then directly use it as embeddings;
                if [B, C], used to do sampling (mu shared by slots)
            slots_log_sigma: if None, no sampling;
                if [B, C], used to do sampling (sigma shared by slots)
            fg_mask: [B, num_slots], boolean mask indicating fg/bg slots
        """
        assert len(slots_mu.shape) == 3 and fg_mask is not None
        bg_mask = ~fg_mask
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = slots_mu
        fg_slots, bg_slots = slots[fg_mask], slots[bg_mask]

        # calculate number of fg slots in each data
        num_fgs = fg_mask.sum(1)  # [B]
        fg_start_idx = [num_fgs[:i].sum().item() for i in range(bs)]
        fg_end_idx = [num_fgs[:i + 1].sum().item() for i in range(bs)]
        num_bgs = (bg_mask).sum(1)  # [B]
        bg_start_idx = [num_bgs[:i].sum().item() for i in range(bs)]
        bg_end_idx = [num_bgs[:i + 1].sum().item() for i in range(bs)]

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            fg_slots_prev = fg_slots
            bg_slots_prev = bg_slots

            # Attention.
            fg_q = self.project_q(fg_slots)
            bg_q = self.bg_project_q(bg_slots)

            logits = torch.empty((bs, self.num_slots, num_inputs)).type_as(k)
            k_trans = k.transpose(2, 1).contiguous()
            for i in range(bs):
                one_fg_q = fg_q[fg_start_idx[i]:fg_end_idx[i]].unsqueeze(0)
                fg_logits = torch.matmul(one_fg_q, k_trans[i:i + 1])
                one_bg_q = bg_q[bg_start_idx[i]:bg_end_idx[i]].unsqueeze(0)
                bg_logits = torch.matmul(one_bg_q, k_trans[i:i + 1])
                logits[i:i + 1] = torch.cat([fg_logits, bg_logits], dim=1)

            attn = F.softmax(logits, dim=-1) + self.epsilon
            # `attn` has shape: [batch_size, num_slots, num_inputs].

            # Weighted mean.
            attn = attn / attn.sum(dim=-1, keepdim=True)
            fg_attn, bg_attn = attn[fg_mask], attn[bg_mask]
            updates = torch.empty(
                (bs, self.num_slots, self.slot_size)).type_as(attn)
            for i in range(bs):
                one_fg_attn = fg_attn[fg_start_idx[i]:fg_end_idx[i]]
                fg_updates = torch.matmul(one_fg_attn.unsqueeze(0), v[i:i + 1])
                one_bg_attn = bg_attn[bg_start_idx[i]:bg_end_idx[i]]
                bg_updates = torch.matmul(one_bg_attn.unsqueeze(0), v[i:i + 1])
                updates[i:i + 1] = torch.cat([fg_updates, bg_updates], dim=1)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N,H)
            # so flatten batch and slots dimension
            fg_slots = self.gru(updates[fg_mask], fg_slots_prev)
            fg_slots = fg_slots + self.mlp(fg_slots)

            bg_slots = self.gru(updates[bg_mask], bg_slots_prev)
            bg_slots = bg_slots + self.mlp(bg_slots)

        slots = torch.empty((bs, self.num_slots, self.slot_size)).type_as(k)
        slots[fg_mask] = fg_slots
        slots[bg_mask] = bg_slots
        return slots


class ObjSlotAttentionModel(SlotAttentionModel):

    def __init__(self,
                 clip_model: CLIP,
                 use_clip_vision: bool,
                 use_clip_text: bool,
                 text2slot_model: nn.Module,
                 resolution: Tuple[int, int],
                 num_slots: int,
                 num_iterations: int,
                 enc_resolution: Tuple[int, int] = ...,
                 enc_channels: int = 3,
                 enc_pos_enc: bool = False,
                 slot_size: int = 64,
                 dec_kernel_size: int = 5,
                 dec_hidden_dims: Tuple[int, ...] = ...,
                 dec_resolution: Tuple[int, int] = ...,
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

        slot_attn = BgSepSlotAttention if \
            self.use_bg_sep_slot else SlotAttention
        # slot_attn = ObjBgSepSlotAttention if \
        #     self.use_bg_sep_slot else ObjSlotAttention
        self.slot_attention = slot_attn(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
        )

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
        return slots, obj_mask

    def encode(self, x):
        """Encode from img to slots."""
        img, text, padding = x['img'], x['text'], x['padding']
        encoder_out = self._get_encoder_out(img)  # transformed vision feature
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        # slot initialization
        slot_mu, obj_mask = self._get_slot_embedding(text, padding)

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(encoder_out, slot_mu, fg_mask=obj_mask)
        return slots


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
                 use_word_set: bool = False,
                 use_padding_mask: bool = False,
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
            use_word_set=use_word_set,
            use_padding_mask=use_padding_mask,
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
