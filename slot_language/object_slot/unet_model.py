import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from unet import UNet, UpBlock
from obj_model import ConcatSoftPositionEmbed, SemPosSepSlotAttention

sys.path.append('../')

from model import SlotAttention, BgSepSlotAttention, SoftPositionEmbed, SlotAttentionModel
from utils import assert_shape, conv_transpose_out_shape


class UNetSlotAttentionModel(SlotAttentionModel):
    """CLIP + Slot Attention with UNet like structure.
    UNet's downsample and upsample lead to stronger representation capacity.
    """

    def __init__(
        self,
        clip_model: CLIP,
        use_clip_vision: bool,
        use_clip_text: bool,
        text2slot_model: nn.Module,  # if None, then don't use it here
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations: int,
        slot_size: int = 64,
        slot_mlp_size: int = 128,
        kernel_size: int = 5,
        enc_channels: Tuple[int, ...] = (64, 64, 64, 64),  # 3 times down-up
        dec_channels: Tuple[int, ...] = (64, 64, 64, 64, 64),  # 4 times up
        enc_pos_enc: bool = False,
        dec_resolution: Tuple[int, int] = (8, 8),
        use_entropy_loss: bool = False,
        use_bg_sep_slot: bool = False,
    ):
        nn.Module.__init__(self)
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_size = slot_size
        self.slot_mlp_size = slot_mlp_size
        self.enc_channels = enc_channels
        self.enc_pos_enc = enc_pos_enc
        self.enc_resolution = resolution
        self.dec_resolution = dec_resolution
        self.use_double_conv = False
        self.use_maxpool = False
        self.use_bilinear = True
        self.use_bn = False
        self.out_features = slot_size

        # encoder output feature maps with channel `enc_channels[0]`
        # which should be equal to `slot_size`
        assert enc_channels[0] == slot_size
        # SlotAttention outputs features with channel `slot_size`
        # which should be equal to decoder in_dim `dec_channels[0]`
        assert slot_size == dec_channels[0]

        # Pre-trained CLIP model, we freeze it here
        self.clip_model = clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.use_clip_vision = use_clip_vision
        self.use_clip_text = use_clip_text

        # Text2Slot that generates slot embedding from text features
        if self.use_clip_text:
            assert text2slot_model is not None
        self.text2slot_model = text2slot_model

        # Build encoder with UNet style
        self.encoder = UNet(
            3,
            enc_channels,
            kernel_size,
            use_double_conv=self.use_double_conv,
            use_maxpool=self.use_maxpool,
            use_bilinear=self.use_bilinear,
            use_bn=self.use_bn)

        # Build Encoder related modules
        if self.enc_pos_enc:
            self.encoder_pos_embedding = SoftPositionEmbed(
                3, self.out_features, self.enc_resolution)
        else:
            self.encoder_pos_embedding = None
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        self._build_decoder(dec_channels, kernel_size)

        # same convolutions
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(
                dec_channels[-1],
                dec_channels[-1],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                dec_channels[-1],
                4,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
            ),
        )

        self.decoder_pos_embedding = SoftPositionEmbed(3, self.out_features,
                                                       self.dec_resolution)

        self.use_bg_sep_slot = use_bg_sep_slot
        slot_attn = BgSepSlotAttention if use_bg_sep_slot else SlotAttention
        self.slot_attention = slot_attn(
            in_features=self.out_features,
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=slot_size,
            mlp_hidden_size=slot_mlp_size,
        )

        self.use_entropy_loss = use_entropy_loss  # -p*log(p)

    def _build_decoder(self, channels, kernel_size):
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        for i in range(len(channels) - 1):
            modules.append(
                UpBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    use_double_conv=self.use_double_conv,
                    use_bilinear=self.use_bilinear,
                    use_bn=self.use_bn))
            out_size = out_size * 2 if self.use_bilinear else \
                conv_transpose_out_shape(
                    out_size, 2, kernel_size // 2, kernel_size, 1)
        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )
        self.decoder = nn.Sequential(*modules)

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

    def decode(self, slots, img_shape):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape
        batch_size, num_channels, height, width = img_shape

        # spatial broadcast
        decoder_in = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        out = self.out_conv(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    @property
    def dtype(self):
        return self.out_conv[0].weight.dtype


class SemPosSepUNetSlotAttentionModel(UNetSlotAttentionModel):

    def __init__(self,
                 clip_model: CLIP,
                 use_clip_vision: bool,
                 use_clip_text: bool,
                 text2slot_model: nn.Module,
                 resolution: Tuple[int, int],
                 num_slots: int,
                 num_iterations: int,
                 slot_size: int = 64,
                 enc_pos_size: int = 8,
                 dec_pos_size: int = None,
                 slot_mlp_size: int = 128,
                 kernel_size: int = 5,
                 enc_channels: Tuple[int, ...] = (64, 64, 64, 64),
                 dec_channels: Tuple[int, ...] = (64, 64, 64, 64, 64),
                 enc_pos_enc: bool = False,
                 dec_resolution: Tuple[int, int] = (8, 8),
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
            slot_size=slot_size,
            slot_mlp_size=slot_mlp_size,
            kernel_size=kernel_size,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            enc_pos_enc=enc_pos_enc,
            dec_resolution=dec_resolution,
            use_entropy_loss=use_entropy_loss,
            use_bg_sep_slot=use_bg_sep_slot)

        self.enc_pos_size = enc_pos_size
        self.dec_pos_size = dec_pos_size

        # Build Encoder related modules
        self.pos_ratio = enc_pos_size / slot_size
        self.encoder_pos_embedding = ConcatSoftPositionEmbed(
            3, int(self.enc_channels * self.pos_ratio), self.enc_resolution)
        del self.encoder_out_layer  # no mixing pos and sem

        self.slot_attention = SemPosSepSlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            pos_dim=self.enc_pos_size,
        )

    def _build_decoder(self, channels, kernel_size):
        if self.dec_pos_size is not None:
            self.decoder_pos_embedding = ConcatSoftPositionEmbed(
                3, self.dec_pos_size, self.dec_resolution)
            channels[0] += self.dec_pos_size
        return super()._build_decoder(channels, kernel_size)
