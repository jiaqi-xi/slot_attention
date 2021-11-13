import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from unet_model import UNetSlotAttentionModel, \
    SlotAttention, BgSepSlotAttention, SoftPositionEmbed


class TwoStreamSlotAttentionModel(UNetSlotAttentionModel):
    """CLIP + Slot Attention with UNet like structure.
    UNet's downsample and upsample lead to stronger representation capacity.
    """

    def __init__(self,
                 clip_model: CLIP,
                 use_clip_vision: bool,
                 use_clip_text: bool,
                 text2slot_model: nn.Module,
                 text2slot_model_conv: nn.Module,
                 resolution: Tuple[int, int],
                 num_slots: int,
                 num_iterations: int,
                 slot_size: int = 64,
                 slot_mlp_size: int = 128,
                 kernel_size: int = 5,
                 enc_channels: Tuple[int, ...] = (64, 64, 64, 64),
                 dec_channels: Tuple[int, ...] = (64, 64),
                 enc_pos_enc: bool = False,
                 enc_pos_enc_conv: bool = False,
                 dec_pos_enc: bool = False,
                 dec_resolution: Tuple[int, int] = (128, 128),
                 spread_hard_mask: bool = False,
                 finetune_mask: bool = False,
                 use_maxpool: bool = True,
                 use_bn: bool = False,
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
            slot_size=slot_size,
            slot_mlp_size=slot_mlp_size,
            kernel_size=kernel_size,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            enc_pos_enc=enc_pos_enc,
            dec_resolution=dec_resolution,
            use_maxpool=use_maxpool,
            use_bilinear=True,
            use_bn=use_bn,
            use_word_set=use_word_set,
            use_padding_mask=use_padding_mask,
            use_entropy_loss=use_entropy_loss,
            use_bg_sep_slot=use_bg_sep_slot)

        self.enc_pos_enc_conv = enc_pos_enc_conv
        self.dec_pos_enc = dec_pos_enc
        if spread_hard_mask:
            assert finetune_mask
        self.spread_hard_mask = spread_hard_mask
        self.finetune_mask = finetune_mask

        if self.dec_pos_enc:
            self.decoder_pos_embedding = SoftPositionEmbed(
                3, self.out_features, self.dec_resolution)
        else:
            self.decoder_pos_embedding = None

        out_dim = 4 if finetune_mask else 3
        self.out_conv = nn.Conv2d(
            dec_channels[-1], out_dim, kernel_size=3, stride=1, padding=1)

        # building the second stream that outputs Conv kernels with features
        self.text2slot_model_conv = text2slot_model_conv

        if self.enc_pos_enc_conv:
            self.encoder_pos_embedding_conv = SoftPositionEmbed(
                3, self.out_features, self.enc_resolution)
        else:
            self.encoder_pos_embedding_conv = None

        self.encoder_out_layer_conv = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        slot_attn = BgSepSlotAttention if use_bg_sep_slot else SlotAttention
        self.slot_attention_conv = slot_attn(
            in_features=self.out_features,
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=slot_size,
            mlp_hidden_size=slot_mlp_size,
        )

    def _build_decoder(self, channels, kernel_size):
        """No up-sampling in the decoder."""
        modules = []
        for i in range(len(channels) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
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
        return encoder_out

    def _get_visual_features(self, encoder_out, pos_enc, enc_out_layer):
        """Add positional encoding and MLP on image feature maps."""
        # `encoder_out` is of shape [B, C, H, W]
        if pos_enc is not None:
            encoder_out = pos_enc(encoder_out)
        img_features = encoder_out.flatten(2, 3).permute(0, 2, 1)
        img_features = enc_out_layer(img_features)  # [B, H*W, C']
        return img_features

    def _encode_text_feature(self, text):
        """Encode text feature using LM in CLIP."""
        if not self.use_clip_text:
            # not generating slots
            return None
        text_features = self.clip_model.encode_text(
            text,
            lin_proj=False,
            per_token_emb=self.use_word_set,
            return_mask=self.use_padding_mask)  # BC or BLC + padding mask
        if self.use_padding_mask:
            text_features, padding_mask = text_features[0].type(self.dtype), \
                text_features[1].type(self.dtype)
            text_features = dict(
                text_features=text_features, padding_mask=padding_mask)
        else:
            text_features = text_features.type(self.dtype)
        return text_features

    def _get_slot_embedding(self, text_features, text2slot_model):
        """Encode text, generate slot embeddings."""
        if text_features is None:
            return None, None
        slot_mu, slot_log_sigma = text2slot_model(text_features)
        return slot_mu, slot_log_sigma

    def forward(self, x):
        torch.cuda.empty_cache()

        feature_maps, slots, slots_conv = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            feature_maps, slots, slots_conv, x['img'].shape)
        return recon_combined, recons, masks, slots

    def encode(self, x):
        """Encode from img to slots."""
        img, text = x['img'], x['text']

        # encoder_out is of shape [B, C, H, W]
        encoder_out = self._get_encoder_out(img)  # transformed vision feature
        img_features = self._get_visual_features(encoder_out,
                                                 self.encoder_pos_embedding,
                                                 self.encoder_out_layer)
        img_features_conv = self._get_visual_features(
            encoder_out, self.encoder_pos_embedding_conv,
            self.encoder_out_layer_conv)

        # slot initialization
        text_features = self._encode_text_feature(text)
        slot_mu, slot_log_sigma = self._get_slot_embedding(
            text_features, self.text2slot_model)
        slot_mu_conv, slot_log_sigma_conv = self._get_slot_embedding(
            text_features, self.text2slot_model_conv)

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(img_features, slot_mu, slot_log_sigma)
        slots_conv = self.slot_attention_conv(img_features_conv, slot_mu_conv,
                                              slot_log_sigma_conv)
        return encoder_out, slots, slots_conv

    def decode(self, feature_maps, slots, slots_conv, img_shape):
        """Decode from slots to reconstructed images and masks.

        Args:
            feature_maps: [B, C, H, W], feature maps from `self.encoder`
            slots/slots_conv: [B, num_slots, C]
        """
        batch_size, num_slots, slot_size = slots.shape
        batch_size, num_channels, height, width = img_shape

        # Conv feature maps to get seg_mask with slots_conv as kernels
        # seg_mask is of shape [B, num_slots, H, W]
        seg_mask = torch.einsum('bnc,bchw->bnhw', [slots_conv, feature_maps])
        seg_mask = F.softmax(seg_mask, dim=1)

        # spread slots to the regions of seg_mask
        if self.spread_hard_mask:
            seg_mask = (seg_mask == seg_mask.max(1)[0].unsqueeze(1)).float()
        decoder_in = torch.einsum('bnc,bnhw->bnchw', [slots, seg_mask])
        decoder_in = decoder_in.flatten(0, 1)  # [B * num_slots, C, H, W]

        # decode results
        if self.dec_pos_enc:
            decoder_in = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(decoder_in)
        out = self.out_conv(out)
        # `out` has shape: [B * num_slots, 3(+1), H, W].

        out = out.view(batch_size, num_slots, -1, height, width)
        recons = out[:, :, :num_channels, :, :]
        if self.finetune_mask:
            masks = out[:, :, -1:, :, :]
            masks = F.softmax(masks, dim=1)
        else:  # already after softmax
            masks = seg_mask.unsqueeze(2)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, seg_mask.unsqueeze(2)

    @property
    def dtype(self):
        return self.out_conv.weight.dtype
