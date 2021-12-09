from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from clip import CLIP
from utils import Tensor, assert_shape, build_grid, conv_transpose_out_shape, \
    conv_bn_relu, deconv_bn_relu


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

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
        self.attn_scale = self.slot_size**-0.5

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

        # TODO: in case we don't use Text2Slot model
        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(
                torch.zeros((1, 1, self.slot_size)),
                gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(
                torch.zeros((1, 1, self.slot_size)),
                gain=nn.init.calculate_gain("linear")),
        )

    def _init_slots(self, batch_size, slots_mu, slots_log_sigma):
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_mu is not None and slots_log_sigma is None:
            # Text2Slot predicts slot embeddings for each slot individually
            assert len(slots_mu.shape) == 3, 'wrong slot embedding shape!'
            slots = slots_mu
        else:
            if slots_mu is not None:
                # Text2Slot predicts shared mu and sigma for slots
                assert slots_log_sigma is not None
                slots_mu = slots_mu.unsqueeze(1)
                slots_log_sigma = slots_log_sigma.unsqueeze(1)
            else:
                # not using Text2Slot
                assert slots_log_sigma is None
                slots_mu = self.slots_mu
                slots_log_sigma = self.slots_log_sigma
            # if in testing mode, fix random seed to get same slot embedding
            if not self.training:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                slots_init = torch.randn(
                    (1, self.num_slots,
                     self.slot_size)).repeat(batch_size, 1, 1)
            # in training mode, sample from Gaussian with mean and std
            else:
                slots_init = torch.randn(
                    (batch_size, self.num_slots, self.slot_size))
            slots_init = slots_init.type_as(slots_mu)
            slots = slots_mu + slots_log_sigma.exp() * slots_init
        return slots

    def forward(self, inputs, slots_mu, slots_log_sigma=None):
        """Forward function.

        Args:
            inputs: [B, N, C], flattened per-pixel features
            slots_mu: if [B, M, C], then directly use it as embeddings;
                if [B, C], used to do sampling (mu shared by slots)
            slots_log_sigma: if None, no sampling;
                if [B, C], used to do sampling (sigma shared by slots)
        """
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self._init_slots(bs, slots_mu, slots_log_sigma).type_as(inputs)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(
                slots)  # Shape: [batch_size, num_slots, slot_size].

            attn_logits = self.attn_scale * torch.matmul(k, q.transpose(2, 1))
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
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class BgSepSlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention.

    The BgSep one processes fg slots and bg slots seperately.
    We assume the last slot is for background.
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
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False))
        # for bg
        self.bg_project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False))

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )
        # for bg
        self.bg_gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.bg_mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # TODO: in case we don't use Text2Slot model
        self.slots_mu = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros((1, 1, self.slot_size)),
                gain=nn.init.calculate_gain("linear")))
        self.slots_log_sigma = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros((1, 1, self.slot_size)),
                gain=nn.init.calculate_gain("linear")))
        # for bg
        self.bg_slots_mu = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros((1, 1, self.slot_size)),
                gain=nn.init.calculate_gain("linear")))
        self.bg_slots_log_sigma = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros((1, 1, self.slot_size)),
                gain=nn.init.calculate_gain("linear")))

    def _init_slots(self, bs, slots_mu, slots_log_sigma):
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_mu is not None and slots_log_sigma is None:
            # Text2Slot predicts slot embeddings for each slot individually
            assert len(slots_mu.shape) == 3, 'wrong slot embedding shape!'
            slots, bg_slots = slots_mu[:, :-1], slots_mu[:, -1:]
        else:
            # TODO: currently not supporting Text2Slot predict distribution
            assert slots_mu is None and slots_log_sigma is None
            # not using Text2Slot
            mu = self.slots_mu.repeat(bs, self.num_slots - 1, 1)
            log_sigma = self.slots_log_sigma.repeat(bs, self.num_slots - 1, 1)
            bg_mu = self.bg_slots_mu.repeat(bs, 1, 1)
            bg_log_sigma = self.bg_slots_log_sigma.repeat(bs, 1, 1)
            # if in testing mode, fix random seed to get same slot embedding
            if not self.training:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                slots_init = torch.randn(
                    (1, self.num_slots - 1, self.slot_size)).repeat(bs, 1, 1)
                bg_slots_init = torch.randn(
                    (1, 1, self.slot_size)).repeat(bs, 1, 1)
            # in training mode, sample from Gaussian with mean and std
            else:
                slots_init = torch.randn_like(mu)
                bg_slots_init = torch.randn_like(bg_mu)
            slots = mu + log_sigma.exp() * slots_init.type_as(mu)
            bg_slots = bg_mu + bg_log_sigma.exp() * bg_slots_init.type_as(mu)
        return slots, bg_slots

    def forward(self, inputs, slots_mu, slots_log_sigma=None):
        """Forward function.

        Args:
            inputs: [B, N, C], flattened per-pixel features
            slots_mu: if [B, M, C], then directly use it as embeddings;
                if [B, C], used to do sampling (mu shared by slots)
            slots_log_sigma: if None, no sampling;
                if [B, C], used to do sampling (sigma shared by slots)
        """
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [batch_size, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        fg_slots, bg_slots = self._init_slots(bs, slots_mu, slots_log_sigma)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            fg_slots_prev = fg_slots
            bg_slots_prev = bg_slots

            # Attention.
            fg_q = self.project_q(fg_slots)
            bg_q = self.bg_project_q(bg_slots)

            fg_logits = self.attn_scale * torch.matmul(k, fg_q.transpose(2, 1))
            bg_logits = self.attn_scale * torch.matmul(k, bg_q.transpose(2, 1))
            logits = torch.cat([fg_logits, bg_logits], dim=-1)
            attn = F.softmax(logits, dim=-1) + self.epsilon
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            fg_attn, bg_attn = attn[..., :-1], attn[..., -1:]
            fg_attn = fg_attn / fg_attn.sum(dim=1, keepdim=True)
            bg_attn = bg_attn / bg_attn.sum(dim=1, keepdim=True)
            fg_updates = torch.matmul(fg_attn.transpose(1, 2), v)
            bg_updates = torch.matmul(bg_attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N,H)
            # so flatten batch and slots dimension
            fg_slots = self.gru(
                fg_updates.reshape(bs * (self.num_slots - 1), self.slot_size),
                fg_slots_prev.reshape(bs * (self.num_slots - 1),
                                      self.slot_size),
            )
            fg_slots = fg_slots.view(bs, self.num_slots - 1, self.slot_size)
            fg_slots = fg_slots + self.mlp(fg_slots)

            bg_slots = self.gru(
                bg_updates.reshape(bs * 1, self.slot_size),
                bg_slots_prev.reshape(bs * 1, self.slot_size),
            )
            bg_slots = bg_slots.view(bs, 1, self.slot_size)
            bg_slots = bg_slots + self.bg_mlp(bg_slots)

        slots = torch.cat([fg_slots, bg_slots], dim=1)
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
            enc_channels=(3, 64, 64, 64, 64),
            enc_resolution=(7, 7),  # output res of encoder
            visual_feats_channels=512,  # output channel of encoder
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=(64, 64, 64, 64, 64),  # 4 times up
            dec_resolution=(7, 7),  # 7 * (2**5) = 224
            dec_norm='',
        ),
        use_word_set: bool = False,
        use_padding_mask: bool = False,
        use_entropy_loss: bool = False,
    ):
        super().__init__()
        self.resolution = resolution

        self.num_slots = slot_dict['num_slots']
        self.num_iterations = slot_dict['num_iterations']
        self.slot_size = slot_dict['slot_size']
        self.slot_mlp_size = slot_dict['slot_mlp_size']
        self.use_bg_sep_slot = slot_dict['use_bg_sep_slot']

        self.out_features = enc_dict['out_features']
        self.kernel_size = enc_dict['kernel_size']
        self.enc_channels = enc_dict['enc_channels']
        self.enc_resolution = enc_dict['enc_resolution']
        self.visual_feats_channels = enc_dict['visual_feats_channels']
        self.enc_norm = enc_dict['enc_norm']

        self.dec_channels = dec_dict['dec_channels']
        self.dec_resolution = dec_dict['dec_resolution']
        self.dec_norm = dec_dict['dec_norm']

        self.use_word_set = use_word_set
        self.use_padding_mask = use_padding_mask

        # Pre-trained CLIP model, we freeze it here
        self.clip_model = clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.use_clip_vision = use_clip_vision
        self.use_clip_text = use_clip_text
        if not self.use_clip_vision:
            self.enc_resolution = self.resolution
            self.visual_feats_channels = self.enc_channels[-1]
            # self.clip_model.visual = None

        # Text2Slot that generates slot embedding from text features
        if self.use_clip_text:
            assert text2slot_model is not None
        self.text2slot_model = text2slot_model

        # extra loss besides reconstruction loss
        self.use_entropy_loss = use_entropy_loss  # -p*log(p)

        self._build_encoder()
        self._build_decoder()
        self._build_slot_attention()

    def _build_slot_attention(self):
        slot_attn = BgSepSlotAttention if self.use_bg_sep_slot else SlotAttention
        self.slot_attention = slot_attn(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
        )

    def _build_encoder(self):
        # we build an encoder as in original Slot Attention paper
        if not self.use_clip_vision:
            # Build Encoder
            self.encoder = nn.Sequential(*[
                conv_bn_relu(
                    self.enc_channels[i],
                    self.enc_channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=1,
                    norm=self.enc_norm)
                for i in range(len(self.enc_channels) - 1)
            ])

        # Build Encoder related modules
        self.encoder_pos_embedding = SoftPositionEmbed(
            3, self.visual_feats_channels, self.enc_resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_feats_channels),  # from SAVi
            nn.Linear(self.visual_feats_channels, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

    def _build_decoder(self):
        # Build Decoder
        assert self.dec_channels[0] == self.slot_size
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        stride = 2
        for i in range(len(self.dec_channels) - 1):
            modules.append(
                deconv_bn_relu(
                    self.dec_channels[i],
                    self.dec_channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=stride,
                    norm=self.dec_norm))
            out_size = conv_transpose_out_shape(out_size, stride, 2, 5,
                                                stride - 1)
            if out_size == self.resolution[0]:
                stride = 1

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Conv2d(
                self.dec_channels[-1],
                4,
                kernel_size=3,
                stride=1,
                padding=1,
            ))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(3, self.slot_size,
                                                       self.dec_resolution)

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        if self.use_clip_vision:
            encoder_out = self.clip_model.encode_image(
                img, global_feats=False, downstream=True)  # BCDD
            encoder_out = encoder_out.type(self.dtype)
        else:
            encoder_out = self.encoder(img)
        img_feats = encoder_out  # Conv features without pos_enc
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, C, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, C, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, C]
        return encoder_out, img_feats

    def _get_slot_embedding(self, text):
        """Encode text, generate slot embeddings."""
        if not self.use_clip_text:
            # not generating slots
            return None, None
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
        slot_mu, slot_log_sigma = self.text2slot_model(text_features)
        return slot_mu, slot_log_sigma, text_features

    def forward(self, x):
        torch.cuda.empty_cache()

        slots, img_feats, text_feats = self.encode(x)

        recon_combined, recons, masks, slots = self.decode(
            slots, x['img'].shape)

        if not self.training:
            return recon_combined, recons, masks, slots
        return recon_combined, recons, masks, slots, img_feats, text_feats

    def encode(self, x):
        """Encode from img to slots."""
        img, text = x['img'], x['text']
        encoder_out, img_feats = self._get_encoder_out(img)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]
        # `img_feats` has shape: [bs, visual_feats_channels, height, width]

        # slot initialization
        slot_mu, slot_log_sigma, text_feats = self._get_slot_embedding(text)

        # (batch_size, self.num_slots, self.slot_size)
        slots = self.slot_attention(encoder_out, slot_mu, slot_log_sigma)
        return slots, img_feats, text_feats

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
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def eval_loss_function(self, input):
        """Loss computation in eval, we only care about reconstruction loss."""
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input['img'])
        loss_dict = {
            'recon_loss': loss,
        }
        return loss_dict

    def loss_function(self, input):
        if not self.training:
            return self.eval_loss_function(input)

        recon_combined, recons, masks, slots, \
            img_feats, text_feats = self.forward(input)
        return self.calc_train_loss(input['img'], recon_combined, recons,
                                    masks, slots, img_feats, text_feats)

    def calc_train_loss(self, img, recon_combined, recons, masks, slots,
                        img_feats, text_feats):
        """Compute loss that are general for SlotAttn models."""
        loss = F.mse_loss(recon_combined, img)
        loss_dict = {
            "recon_loss": loss,
        }
        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy_loss'] = entropy_loss
        return loss_dict

    @property
    def dtype(self):
        return self.decoder[0][0].weight.dtype


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
