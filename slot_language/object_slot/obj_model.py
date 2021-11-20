import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from obj_utils import SepLinear, SepLayerNorm, SepGRUCell

sys.path.append('../')

from model import SlotAttention, BgSepSlotAttention, \
    SlotAttentionModel, SoftPositionEmbed


class ObjSlotAttention(SlotAttention):
    """A wrapper for SlotAttention to make forward interface consistent."""

    def forward(self, inputs, slots_mu, slots_log_sigma=None, fg_mask=None):
        return super().forward(inputs, slots_mu, slots_log_sigma)


class SemPosSepSlotAttention(SlotAttention):
    """SlotAttention that treats semantic and position information separately.

    The forward pass is the same, simply replacing Modules with SepModules.

    Args:
        pos_dim: number of dims for position information, w.r.t. `slot_size`.
            E.g., if slot_size = 64 and pos_dim = 8, then if in_features = 144,
                we assume the last 16 channels of the img_feats is pos_enc.
    """

    def __init__(self,
                 in_features,
                 num_iterations,
                 num_slots,
                 slot_size,
                 mlp_hidden_size,
                 pos_dim=8,
                 epsilon=1e-6):
        nn.Module.__init__(self)

        self.pos_ratio = pos_dim / slot_size
        self.pos_dim = pos_dim
        self.in_features = int(in_features * (1 + self.pos_ratio))
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = int(slot_size * (1 + self.pos_ratio))
        self.mlp_hidden_size = int(mlp_hidden_size * (1 + self.pos_ratio))
        self.epsilon = epsilon
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = SepLayerNorm(in_features, self.in_features)
        self.norm_slots = SepLayerNorm(slot_size, self.slot_size)
        self.norm_mlp = SepLayerNorm(slot_size, self.slot_size)

        # Linear maps for the attention module.
        self.project_q = SepLinear(
            slot_size, self.slot_size, self.slot_size, bias=False)
        self.project_k = SepLinear(
            in_features, self.in_features, self.slot_size, bias=False)
        self.project_v = SepLinear(
            in_features, self.in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = SepGRUCell(slot_size, self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            SepLinear(slot_size, self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            SepLinear(mlp_hidden_size, self.mlp_hidden_size, self.slot_size),
        )

        # FC to keep the output slot_size and mix sem and pos information
        self.out_mlp = nn.Sequential(
            nn.Linear(self.slot_size, slot_size),
            nn.ReLU(),
            nn.Linear(slot_size, slot_size),
        )

    def _init_slots(self, batch_size, slots_mu, slots_log_sigma):
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        assert len(slots_mu.shape) == 3, 'wrong slot embedding shape!'
        assert int(slots_mu.shape[-1] * (1 + self.pos_ratio)) == self.slot_size
        # pad it with pos_emb, inited as all zeros vector
        slots = torch.cat([
            slots_mu,
            torch.zeros(batch_size, self.num_slots,
                        self.pos_dim).type_as(slots_mu).detach()
        ],
                          dim=-1)
        return slots

    def forward(self, inputs, slots_mu, slots_log_sigma=None, fg_mask=None):
        slots = super().forward(inputs, slots_mu, slots_log_sigma)
        return self.out_mlp(slots)


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
            use_word_set=False,
            use_padding_mask=False,
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


class SemPosSepObjSlotAttentionModel(ObjSlotAttentionModel):

    def __init__(self,
                 clip_model: CLIP,
                 use_clip_vision: bool,
                 use_clip_text: bool,
                 text2slot_model: nn.Module,
                 resolution: Tuple[int, int],
                 num_slots: int,
                 num_iterations: int,
                 enc_resolution: Tuple[int, int] = (128, 128),
                 enc_channels: int = 3,
                 slot_size: int = 64,
                 enc_pos_size: int = 8,
                 dec_pos_size: int = None,
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
            enc_pos_enc=True,
            slot_size=slot_size,
            dec_kernel_size=dec_kernel_size,
            dec_hidden_dims=dec_hidden_dims,
            dec_resolution=dec_resolution,
            slot_mlp_size=slot_mlp_size,
            use_entropy_loss=use_entropy_loss,
            use_bg_sep_slot=use_bg_sep_slot)

        self.enc_pos_size = enc_pos_size
        self.dec_pos_size = dec_pos_size

        # Build Encoder related modules
        self.pos_ratio = enc_pos_size / slot_size
        self.encoder_pos_embedding = ConcatSoftPositionEmbed(
            3, int(self.enc_channels * self.pos_ratio), self.enc_resolution)
        del self.encoder_out_layer  # no mixing pos and sem

        # build Decoder related modules
        self._build_decoder()

        self.slot_attention = SemPosSepSlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            pos_dim=self.enc_pos_size,
        )

    def _build_decoder(self):
        # Build Decoder
        if self.dec_pos_size is not None:
            self.decoder_pos_embedding = ConcatSoftPositionEmbed(
                3, self.dec_pos_size, self.dec_resolution)
            self.dec_hidden_dims[-1] += self.dec_pos_size

        modules = []
        for i in range(len(self.dec_hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.dec_hidden_dims[i],
                        self.dec_hidden_dims[i - 1],
                        kernel_size=self.dec_kernel_size,
                        stride=2,
                        padding=self.dec_kernel_size // 2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ))

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features,
                    self.out_features,
                    kernel_size=self.dec_kernel_size,
                    stride=1,
                    padding=self.dec_kernel_size // 2,
                    output_padding=0,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    self.out_features,
                    4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0,
                ),
            ))

        self.decoder = nn.Sequential(*modules)

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        if self.use_clip_vision:
            encoder_out = self.clip_model.encode_image(
                img, global_feats=False, downstream=True)  # BCDD
            encoder_out = encoder_out.type(self.dtype)
        else:
            encoder_out = self.encoder(img)
        encoder_out = self.encoder_pos_embedding(encoder_out).\
            permute(0, 2, 3, 1).flatten(1, 2)
        return encoder_out  # [B, H*W, C]


class ConcatSoftPositionEmbed(SoftPositionEmbed):
    """Concat along channel dim."""

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return torch.cat(
            [inputs, emb_proj.repeat(inputs.shape[0], 1, 1, 1)], dim=1)
