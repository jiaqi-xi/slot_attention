import sys
from typing import Tuple

import torch
from torch import nn

from clip import CLIP
from unet import UNet
from resnet import StackedResBlock
from obj_utils import SepLinear, SepLayerNorm, SepGRUCell

sys.path.append('../')

from model import SlotAttention, BgSepSlotAttention, \
    SlotAttentionModel, SoftPositionEmbed


class ObjSlotAttention(SlotAttention):
    """A wrapper for SlotAttention."""

    def forward(self, inputs, slots_mu, slots_log_sigma=None):
        return super().forward(inputs, slots_mu, slots_log_sigma)


class ObjBgSepSlotAttention(BgSepSlotAttention):
    """A wrapper for BgSepSlotAttention."""

    def forward(self, inputs, slots_mu, slots_log_sigma=None):
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

    def forward(self, inputs, slots_mu, slots_log_sigma=None):
        # [B, num_slots, C]
        slots = super().forward(inputs, slots_mu, slots_log_sigma)
        slot_size = int(self.slot_size / (1 + self.pos_ratio))
        assert slots.shape[-1] - slot_size == self.pos_dim
        sem_slots, pos_slots = slots[..., :slot_size], slots[..., slot_size:]
        return self.out_mlp(slots), sem_slots, pos_slots


class SemPosBgSepSlotAttention(BgSepSlotAttention):

    def __init__(self,
                 in_features,
                 num_iterations,
                 num_slots,
                 slot_size,
                 mlp_hidden_size,
                 pos_dim=8,
                 epsilon=0.000001):
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

        # Linear maps for the attention module.
        self.project_k = SepLinear(
            in_features, self.in_features, self.slot_size, bias=False)
        self.project_v = SepLinear(
            in_features, self.in_features, self.slot_size, bias=False)
        self.project_q = nn.Sequential(
            SepLayerNorm(slot_size, self.slot_size),
            SepLinear(slot_size, self.slot_size, self.slot_size, bias=False))
        # for bg
        self.bg_project_q = nn.Sequential(
            SepLayerNorm(slot_size, self.slot_size),
            SepLinear(slot_size, self.slot_size, self.slot_size, bias=False))

        # Slot update functions.
        self.gru = SepGRUCell(slot_size, self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            SepLayerNorm(slot_size, self.slot_size),
            SepLinear(slot_size, self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            SepLinear(mlp_hidden_size, self.mlp_hidden_size, self.slot_size),
        )
        # for bg
        self.bg_gru = SepGRUCell(slot_size, self.slot_size, self.slot_size)
        self.bg_mlp = nn.Sequential(
            SepLayerNorm(slot_size, self.slot_size),
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
        # for bg
        self.bg_out_mlp = nn.Sequential(
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
        return slots[:, :-1], slots[:, -1:]

    def forward(self, inputs, slots_mu, slots_log_sigma=None):
        # [B, num_slots, C]
        slots = super().forward(inputs, slots_mu, slots_log_sigma)
        slot_size = int(self.slot_size / (1 + self.pos_ratio))
        assert slots.shape[-1] - slot_size == self.pos_dim
        sem_slots, pos_slots = slots[..., :slot_size], slots[..., slot_size:]
        slots = torch.cat(
            [self.out_mlp(slots[:, :-1]),
             self.bg_out_mlp(slots[:, -1:])],
            dim=1)
        return slots, sem_slots, pos_slots


class ObjSlotAttentionModel(SlotAttentionModel):

    def __init__(
        self,
        clip_model: CLIP,
        use_clip_vision: bool,
        use_clip_text: bool,
        text2slot_model: nn.Module,
        resolution: Tuple[int, int],
        slot_dict=dict(
            num_slots=7,
            num_iterations=3,
            slot_size=64,
            slot_mlp_size=128,
            use_bg_sep_slot=False,
        ),
        enc_dict=dict(
            out_features=64,
            kernel_size=5,
            use_unet=False,
            use_resnet=False,
            enc_channels=(3, 64, 64, 64, 64),
            enc_resolution=(7, 7),  # output res of encoder
            visual_feats_channels=512,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=(64, 64, 64, 64, 64),  # 4 times up
            dec_resolution=(7, 7),  # 7 * (2**5) = 224
            dec_norm='',
        ),
        use_entropy_loss: bool = False,
    ):
        self.use_unet = enc_dict['use_unet']
        self.use_resnet = enc_dict['use_resnet']
        assert not (self.use_resnet and self.use_unet)
        if self.use_unet or self.use_resnet:
            assert not use_clip_vision

        super().__init__(
            clip_model,
            use_clip_vision,
            use_clip_text,
            text2slot_model,
            resolution,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            use_word_set=False,
            use_padding_mask=False,
            use_entropy_loss=use_entropy_loss,
        )

    def _build_slot_attention(self):
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

    def _build_encoder(self):
        if self.use_unet:
            print('Use UNet as encoder!')
            self.visual_feats_channels = self.enc_channels[1]
            self.encoder = UNet(
                self.enc_channels[0],
                self.enc_channels[1:],
                self.kernel_size,
                use_double_conv=False,
                use_maxpool=True,
                use_bilinear=True,
                norm=self.enc_norm)
            self.encoder_pos_embedding = SoftPositionEmbed(
                3, self.visual_feats_channels, self.enc_resolution)
            self.encoder_out_layer = nn.Sequential(
                nn.LayerNorm(self.visual_feats_channels),  # from SAVi
                nn.Linear(self.visual_feats_channels, self.out_features),
                nn.ReLU(),
                nn.Linear(self.out_features, self.out_features),
            )
        elif self.use_resnet:
            print('Use ResNet as encoder!')
            self.encoder = StackedResBlock(
                self.enc_channels[0],
                self.enc_channels[1:],
                self.kernel_size,
                norm=self.enc_norm)
        else:
            super()._build_encoder()

    def _get_slot_embedding(self, tokens):
        """Encode text, generate slot embeddings.

        Args:
            tokens: [B, N, C]
        """
        if not self.use_clip_text:
            # not generating slots
            return None, None, None
        # we treat each obj as batch dim and get global text (for each phrase)
        bs = tokens.shape[0]
        assert tokens.shape[1] == self.num_slots
        text_features = self.clip_model.encode_text(
            tokens.flatten(0, 1),
            lin_proj=False,
            per_token_emb=False,
            return_mask=False).unflatten(0, (bs, self.num_slots))  # [B, N, C]
        text_features = text_features.type(self.dtype)
        slots, _ = self.text2slot_model(text_features)
        return slots, _, text_features


class SemPosSepObjSlotAttentionModel(ObjSlotAttentionModel):

    def __init__(
        self,
        clip_model: CLIP,
        use_clip_vision: bool,
        use_clip_text: bool,
        text2slot_model: nn.Module,  # if None, then don't use it here
        resolution: Tuple[int, int],
        slot_dict=dict(
            num_slots=7,
            num_iterations=3,
            slot_size=64,
            slot_mlp_size=128,
            use_bg_sep_slot=False,
        ),
        enc_dict=dict(
            out_features=64,
            kernel_size=5,
            enc_pos_size=64,
            use_unet=False,
            use_resnet=False,
            enc_channels=(3, 64, 64, 64, 64),
            enc_resolution=(7, 7),  # output res of encoder
            visual_feats_channels=512,  # output channel of encoder
            enc_norm='',
        ),
        dec_dict=dict(
            dec_pos_size=None,
            dec_channels=(64, 64, 64, 64, 64),  # 4 times up
            dec_resolution=(7, 7),  # 7 * (2**5) = 224
            dec_norm='',
        ),
        use_entropy_loss: bool = False,
    ):
        self.enc_pos_size = enc_dict['enc_pos_size']
        self.dec_pos_size = dec_dict['dec_pos_size']

        super().__init__(
            clip_model,
            use_clip_vision,
            use_clip_text,
            text2slot_model,
            resolution,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            use_entropy_loss=use_entropy_loss,
        )

    def _build_slot_attention(self):
        slot_attn = SemPosBgSepSlotAttention if \
            self.use_bg_sep_slot else SemPosSepSlotAttention
        self.slot_attention = slot_attn(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            pos_dim=self.enc_pos_size,
        )

    def _build_encoder(self):
        super()._build_encoder()

        # Build Encoder related modules
        self.pos_ratio = self.enc_pos_size / self.slot_size
        self.encoder_pos_embedding = ConcatSoftPositionEmbed(
            3, int(self.visual_feats_channels * self.pos_ratio),
            self.enc_resolution)
        self.encoder_out_layer = None  # no mixing pos and sem

    def _build_decoder(self):
        # Build Decoder
        if self.dec_pos_size is not None:
            self.dec_channels[0] += self.dec_pos_size

        super()._build_decoder()

        if self.dec_pos_size is not None:
            self.decoder_pos_embedding = ConcatSoftPositionEmbed(
                3, self.dec_pos_size, self.dec_resolution)

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        if self.use_clip_vision:
            encoder_out = self.clip_model.encode_image(
                img, global_feats=False, downstream=True)  # BCDD
            encoder_out = encoder_out.type(self.dtype)
        else:
            encoder_out = self.encoder(img)
        img_feats = encoder_out  # Conv features without pos_enc
        encoder_out = self.encoder_pos_embedding(encoder_out).\
            permute(0, 2, 3, 1).flatten(1, 2)
        return encoder_out, img_feats  # [B, H*W, C]

    def encode(self, x):
        """Encode from img to slots."""
        img, text = x['img'], x['text']
        encoder_out, img_feats = self._get_encoder_out(img)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        # slot initialization
        slot_mu, _, text_feats = self._get_slot_embedding(text)
        # `slot_mu` has shape [batch_size, num_slots, slot_size]

        # (batch_size, self.num_slots, self.slot_size)
        slots, sem_slots, pos_slots = self.slot_attention(encoder_out, slot_mu)
        return slots, sem_slots, pos_slots, img_feats, text_feats

    def forward(self, x):
        torch.cuda.empty_cache()

        slots, sem_slots, pos_slots, img_feats, text_feats = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            slots, x['img'].shape)

        if not self.training:
            return recon_combined, recons, masks, slots
        return recon_combined, recons, masks, (slots, sem_slots, pos_slots), \
            img_feats, text_feats


class ConcatSoftPositionEmbed(SoftPositionEmbed):
    """Concat along channel dim."""

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return torch.cat(
            [inputs, emb_proj.repeat(inputs.shape[0], 1, 1, 1)], dim=1)
