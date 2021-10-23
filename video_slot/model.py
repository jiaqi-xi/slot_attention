from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

import lpips
from utils import Tensor, assert_shape, build_grid, conv_transpose_out_shape


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention.

    Args:
        learnable_slot (bool): whether the mu and sigma of slots are trainable
    """

    def __init__(self,
                 in_features,
                 num_iterations,
                 num_slots,
                 slot_size,
                 mlp_hidden_size,
                 learnable_slot=False,
                 epsilon=1e-6):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.learnable_slot = learnable_slot
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

        slot_init_func = self.register_parameter if \
            learnable_slot else self.register_buffer
        # train the mean and std of slot embedding
        slot_init_func(
            "slots_mu",
            torch.nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros((1, 1, self.slot_size)),
                    gain=nn.init.calculate_gain("linear"))),
        )
        slot_init_func(
            "slots_log_sigma",
            torch.nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros((1, 1, self.slot_size)),
                    gain=nn.init.calculate_gain("linear"))),
        )

    def forward(self, inputs: Tensor, slots_prev=None):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)

        if slots_prev is None:
            # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
            # sample from Gaussian with learned mean and std
            slots_init = torch.randn(
                (batch_size, self.num_slots, self.slot_size))
            slots_init = slots_init.type_as(inputs)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        else:
            # directly use the provided previous slots
            slots = slots_prev

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

    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations: int,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (8, 8),
        use_deconv: bool = True,
        empty_cache: bool = False,
        slot_mlp_size: int = 128,
        learnable_slot: bool = False,
        use_entropy_loss: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.use_deconv = use_deconv
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                ))
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels,
                                                       self.out_features,
                                                       resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    self.upsample2x(self.hidden_dims[i],
                                    self.hidden_dims[i - 1]),
                    nn.ReLU(),
                ))
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # same convolutions
        conv_module = nn.ConvTranspose2d if self.use_deconv else nn.Conv2d
        modules.append(
            nn.Sequential(
                conv_module(
                    self.out_features,
                    self.out_features,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                ),
                nn.ReLU(),
                conv_module(
                    self.out_features,
                    4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels,
                                                       self.out_features,
                                                       self.decoder_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=slot_mlp_size,
            learnable_slot=learnable_slot,
        )

        self.use_entropy_loss = use_entropy_loss  # -p*log(p)

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(encoder_out)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        # spatial broadcast
        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0],
                                  self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, C, H, W]
        masks = out[:, :, -1:, :, :]  # [B, num_slots, 1, H, W]
        masks = F.softmax(masks, dim=1)
        slot_recons = recons * masks + (1 - masks)
        recon_combined = torch.sum(recons * masks, dim=1)
        # recon_combined: [B, C, H, W]
        # recons: [B, num_slots, C, H, W]
        # masks: [B, num_slots, 1, H, W]
        # slot_recons: [B, num_slots, C, H, W]
        # TODO: I return slot_recons instead of slots here!
        # TODO: this is different from the other slot-att models
        return recon_combined, recons, masks, slot_recons

    def upsample2x(self, in_channels, out_channels):
        if self.use_deconv:
            return nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
                output_padding=1,
            )
        else:
            return nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                ),
            )

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        output_dict = {
            'recon_loss': loss,
            'masks': masks,
            'slots': slots,
        }
        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entroly_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            output_dict['entropy'] = entroly_loss
        return output_dict


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


def conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias)


def conv_block(in_channels, out_channels, kernel_size, stride, bias, bn):
    if bn:
        return nn.Sequential(
            conv(in_channels, out_channels, kernel_size, stride, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride, bias),
        nn.ReLU(),
    )


def deconv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    """Output shape could be in_shape * stride"""
    padding = kernel_size // 2
    out_padding = int(stride + 2 * padding - kernel_size)
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=out_padding,
        bias=bias)


def deconv_block(in_channels, out_channels, kernel_size, stride, bias, bn):
    if bn:
        return nn.Sequential(
            deconv(in_channels, out_channels, kernel_size, stride, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    return nn.Sequential(
        deconv(in_channels, out_channels, kernel_size, stride, bias),
        nn.ReLU(),
    )


class ConvAutoEncoder(nn.Module):
    """Predictor given object_t output object_{t+1}"""

    def __init__(self,
                 num_slots: int,
                 in_channels: int,
                 enc_channels: Tuple[int] = [32, 32, 32],
                 dec_channels: Tuple[int] = [32, 32, 32],
                 kernel_size: int = 5,
                 strides: Tuple[int] = [2, 2, 2],
                 use_bn: bool = False,
                 use_softmax: bool = False):
        super().__init__()

        self.num_slots = num_slots
        # if predict mask, then should apply softmax
        self.use_softmax = use_softmax

        assert len(enc_channels) == len(strides) == len(dec_channels)
        encoder_resolution = 128
        decoder_resolution = encoder_resolution
        for stride in strides:
            decoder_resolution //= stride

        enc_channels.insert(0, in_channels)
        encoder = []
        for i in range(len(enc_channels) - 1):
            encoder.append(
                conv_block(
                    enc_channels[i],
                    enc_channels[i + 1],
                    kernel_size,
                    strides[i],
                    bias=not use_bn,
                    bn=use_bn))
        self.encoder = nn.Sequential(*encoder)

        in_size = decoder_resolution
        out_size = in_size
        strides = strides[::-1]  # revert for decoder
        dec_channels.insert(0, enc_channels[-1])
        decoder = []
        for i in range(len(dec_channels) - 1):
            stride = strides[i]
            padding = kernel_size // 2
            out_padding = int(stride + 2 * padding - kernel_size)
            decoder.append(
                deconv_block(
                    dec_channels[i],
                    dec_channels[i + 1],
                    kernel_size,
                    strides[i],
                    bias=not use_bn,
                    bn=use_bn))
            out_size = conv_transpose_out_shape(out_size, stride, padding,
                                                kernel_size, out_padding)

        assert out_size == encoder_resolution, \
            f'ConvAE decoder shape {out_size} does not match {encoder_resolution}!'

        # output convolutions
        decoder.append(
            nn.ConvTranspose2d(
                dec_channels[-1],
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
            ))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        """Predict x at next time step.
        x: [B, C, H, W]
        """
        _, C, H, W = x.shape
        feats = self.encoder(x)
        pred_x = self.decoder(feats)
        if self.use_softmax:
            pred_x = F.softmax(pred_x.view(-1, self.num_slots, C, H, W), dim=1)
            pred_x = pred_x.view(-1, C, H, W)
        return pred_x

    def loss_function(self, x_prev, x_future):
        """x_prev and x_future are of same shape.
        Can be either mask or recon or mask * recon
        """
        x_pred = self.forward(x_prev)
        loss = F.mse_loss(x_pred, x_future)
        return {
            'pred_loss': loss,
            'pred': x_pred,
        }


class PerceptualLoss(nn.Module):

    def __init__(self, arch='vgg'):
        super().__init__()

        assert arch in ['alex', 'vgg', 'squeeze']
        self.loss_fn = lpips.LPIPS(net=arch).eval()
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    def forward(self, x_prev):
        """Just for backward compatibility with AEPredictor"""
        return x_prev

    def loss_function(self, x_prev, x_future):
        """x_prev and x_future are of same shape.
        Should be mask * recon + (1 - mask)
        """
        assert len(x_prev.shape) == len(x_future.shape) == 4
        x_prev = torch.clamp(x_prev, min=-1., max=1.)
        x_future = torch.clamp(x_future, min=-1., max=1.)
        loss = self.loss_fn(x_prev, x_future).mean()
        return {
            'pred_loss': loss,
            'pred': x_prev,
        }
