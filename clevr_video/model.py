from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

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
                 learnable_slot=False,
                 slot_agnostic=True,
                 random_slot=True,
                 epsilon=1e-6):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.learnable_slot = learnable_slot
        self.slot_agnostic = slot_agnostic
        self.random_slot = random_slot
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

        trainable_slot_num = 1 if self.slot_agnostic else self.num_slots
        slot_init_func = self.register_parameter if \
            learnable_slot else self.register_buffer
        if self.random_slot:
            # train the mean and std of slot embedding
            slot_init_func(
                "slots_mu",
                torch.nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.zeros((1, trainable_slot_num, self.slot_size)),
                        gain=nn.init.calculate_gain("linear"))),
            )
            slot_init_func(
                "slots_log_sigma",
                torch.nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.zeros((1, trainable_slot_num, self.slot_size)),
                        gain=nn.init.calculate_gain("linear"))),
            )
        else:
            # train slot embedding itself
            # should definitely be one trainable embedding for each slot
            assert not slot_agnostic, 'cannot use the same emb for each slot!'
            slot_init_func(
                "slots_mu",
                torch.nn.Parameter(
                    nn.init.xavier_normal_(  # TODO: mind the init method here?
                        torch.zeros((1, self.num_slots, self.slot_size)),
                        gain=nn.init.calculate_gain("linear"))),
            )

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if self.random_slot:
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
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        else:
            # use the learned embedding itself, no sampling, no randomness
            slots = self.slots_mu.repeat(batch_size, 1, 1)

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
        empty_cache: bool = False,
        use_relu: bool = False,  # TODO: official code use ReLU
        slot_mlp_size: int = 128,
        learnable_slot: bool = False,
        slot_agnostic: bool = True,
        random_slot: bool = True,
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
                    nn.ReLU() if use_relu else nn.LeakyReLU(),
                ))
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels,
                                                       self.out_features,
                                                       resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU() if use_relu else nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU() if use_relu else nn.LeakyReLU(),
                ))
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
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
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    output_padding=0,
                ),
                nn.ReLU() if use_relu else nn.LeakyReLU(),
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
            slot_agnostic=slot_agnostic,
            random_slot=random_slot,
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
