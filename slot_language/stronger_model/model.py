import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from unet import UNetEncoder, UNetDecoder

sys.path.append('../')

from model import SlotAttention, SoftPositionEmbed, SlotAttentionModel


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
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),  # down-up 3 times
        enc_pos_enc: bool = False,  # because CLIP's vision encoder already has?
        dec_resolution: Tuple[int, int] = (16, 16),
        use_double_conv: bool = False,
        use_maxpool: bool = True,
        use_bilinear: bool = True,
        use_bn: bool = False,
        use_word_set: bool = False,
        use_padding_mask: bool = False,
        use_entropy_loss: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.down_scale = 2**(len(hidden_dims) - 1)
        self.enc_resolution = tuple(
            [res // self.down_scale for res in self.resolution])
        self.enc_pos_enc = enc_pos_enc
        self.dec_resolution = dec_resolution
        self.use_word_set = use_word_set
        self.use_padding_mask = use_padding_mask
        self.out_features = hidden_dims[-1]

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
        self.encoder = UNetEncoder(
            3,
            hidden_dims,
            kernel_size,
            use_double_conv=use_double_conv,
            use_maxpool=use_maxpool,
            use_bilinear=use_bilinear,
            use_bn=use_bn)

        # Build Encoder related modules
        if self.enc_pos_enc:
            self.encoder_pos_embedding = SoftPositionEmbed(
                3, hidden_dims[-1], self.enc_resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder with UNet style
        self.decoder = UNetDecoder(
            hidden_dims,
            kernel_size,
            use_double_conv=use_double_conv,
            use_bilinear=use_bilinear,
            use_bn=use_bn)

        # same convolutions
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(
                self.out_features,
                self.out_features,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
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
        )

        self.decoder_pos_embedding = SoftPositionEmbed(3, self.out_features,
                                                       self.dec_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=slot_size,
            mlp_hidden_size=slot_mlp_size,
        )

        self.use_entropy_loss = use_entropy_loss  # -p*log(p)

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
