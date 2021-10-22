from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from text_model import Text2Slot
from utils import Tensor, assert_shape, build_grid, conv_transpose_out_shape


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention.

    Args:
        slot_agnostic (bool): If True, all slots share trained embedding.
            If False, we train embeddings seperately for each slot.
            Defaults to True (as in the paper).
        random_slot (bool): If True, we train mu and sigma for slot embedding,
            and sample slot from the Gaussian when forward pass. If False, we
            train slot embedding itself (similar to the learnable positional
            embedding in DETR), so that we use the same embedding to interact
            with input image features. Defaults to True (as in the paper).
    """

    def __init__(self,
                 in_features,
                 num_iterations,
                 num_slots,
                 slot_size,
                 mlp_hidden_size,
                 epsilon=1e-6):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs: Tensor, slots_mu: Tensor, slots_log_sigma=None):
        """Forward function.

        Args:
            inputs: [B, N, C], flattened per-pixel features
            slots_mu: if [B, M, C], then directly use it as embeddings;
                if [B, C], used to do sampling (mu shared by slots)
            slots_log_sigma: if None, no sampling;
                if [B, C], used to do sampling (sigma shared by slots)
        """
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_log_sigma is None:
            assert len(slots_mu.shape) == 3, 'wrong slot embedding shape!'
            slots = slots_mu
        else:
            # if in testing mode, fix random seed to get same slot embedding
            if not self.training:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                slots_init = torch.randn(
                    (1, self.num_slots,
                     self.slot_size)).repeat(batch_size, 1, 1)
            # in training mode, sample from Gaussian with learned mean and std
            else:
                slots_init = torch.randn(
                    (batch_size, self.num_slots, self.slot_size))
            slots_init = slots_init.type_as(inputs)
            slots = slots_mu + slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(
                slots)  # Shape: [batch_size, num_slots, slot_size].

            attn_norm_factor = self.slot_size**-0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N,H)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SlotAttentionModel(nn.Module):
    """CLIP + Slot Attention.

    CLIP extracts vision and text feature from input image-text pairs.
    Text2Slot module generates slot embedding for slot initialization.
    Slot Attention module performs iterative clustering and reconstruction.
    """

    def __init__(
        self,
        clip_model: CLIP,
        text2slot_model: Text2Slot,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations: int,
        enc_channels: int = 3,  # output channel of pre-trained encoder
        enc_global_feats: bool = False,  # should use patch features?
        enc_pos_enc: bool = False,  # because CLIP's vision encoder already has?
        slot_size: int = 64,
        dec_kernel_size: int = 3,
        dec_hidden_dims: Tuple[int, ...] = (64, 64, 64, 64, 64),
        dec_resolution: Tuple[int, int] = (7, 7),  # 7 * (2**5) = 224
        empty_cache: bool = False,
        slot_mlp_size: int = 128,
        use_entropy_loss: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.enc_channels = enc_channels
        self.enc_global_feats = enc_global_feats
        self.enc_pos_enc = enc_pos_enc
        self.dec_kernel_size = dec_kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.dec_hidden_dims = dec_hidden_dims
        self.dec_resolution = dec_resolution
        self.slot_mlp_size = slot_mlp_size
        self.out_features = self.dec_hidden_dims[-1]

        # Pre-trained CLIP model, we freeze it here
        self.clip_model = clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Text2Slot that generates slot embedding from text features
        self.text2slot_model = text2slot_model

        # Build Encoder related modules
        if self.enc_pos_enc:
            self.encoder_pos_embedding = SoftPositionEmbed(
                3, self.enc_channels, self.resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.enc_channels, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = dec_resolution[0]
        out_size = in_size

        for i in range(len(self.dec_hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.dec_hidden_dims[i],
                        self.dec_hidden_dims[i - 1],
                        kernel_size=self.dec_kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ))
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features,
                    self.out_features,
                    kernel_size=self.dec_kernel_size,
                    stride=1,
                    padding=2,
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
        self.decoder_pos_embedding = SoftPositionEmbed(3, self.out_features,
                                                       self.dec_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
        )

        self.use_entropy_loss = use_entropy_loss  # -p*log(p)

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        encoder_out = self.clip_model.encode_image(
            img, global_feats=self.enc_global_feats, downstream=True)  # BCDD
        # may not applying pos_enc because Encoder in CLIP already does so
        if self.enc_pos_enc:
            encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, C, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, C, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, C]
        return encoder_out

    def _get_slot_embedding(self, text):
        """Encode text, generate slot embeddings."""
        text_features = self.clip_model.encode_text(text, lin_proj=False)  # BC
        slot_mu, slot_log_sigma = self.text2slot_model(text_features)
        return slot_mu, slot_log_sigma

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        img, text = x['img'], x['text']
        batch_size, num_channels, height, width = img.shape
        encoder_out = self._get_encoder_out(img)  # transformed vision feature
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        # slot initialization
        slot_mu, slot_log_sigma = self._get_slot_embedding(text)

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(encoder_out, slot_mu, slot_log_sigma)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        # spatial broadcast
        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(1, 1, self.dec_resolution[0],
                                  self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        loss_dict = {
            "loss": loss,
        }
        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entroly_loss = (-masks * torch.log(masks + 1e-6)).sum(1)[0].mean()
            loss_dict['entropy'] = entroly_loss
        return loss_dict


class SoftPositionEmbed(nn.Module):

    def __init__(self, num_channels: int, hidden_size: int,
                 resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(
            in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return inputs + emb_proj
