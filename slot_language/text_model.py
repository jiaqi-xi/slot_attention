from typing import Tuple

import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from utils import Tensor


def fc_bn_relu(in_dim, out_dim, use_bn):
    if use_bn:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.ReLU(),
    )


def build_mlps(in_channels, hidden_sizes, out_channels, use_bn):
    if hidden_sizes is None or len(hidden_sizes) == 0:
        return nn.Linear(in_channels, out_channels)
    modules = [fc_bn_relu(in_channels, hidden_sizes[0], use_bn=use_bn)]
    for i in range(0, len(hidden_sizes) - 1):
        modules.append(
            fc_bn_relu(hidden_sizes[i], hidden_sizes[i + 1], use_bn=use_bn))
    modules.append(nn.Linear(hidden_sizes[-1], out_channels))
    return nn.Sequential(*modules)


class MLPText2Slot(nn.Module):
    """Generate slot embedding from text features using MLPs.

    Args:
        in_channels (int): channels of input text features.
        hidden_sizes (Tuple[int]): MLPs hidden sizes.
        predict_dist (bool): whether to predict the (shared) mu and log_sigma
            of slot embedding, or directly predict each slot's value.
    """

    def __init__(self,
                 in_channels: int,
                 num_slots: int,
                 slot_size: int,
                 hidden_sizes: Tuple[int] = (256, ),
                 predict_dist: bool = True,
                 use_bn: bool = True):
        super(MLPText2Slot, self).__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.predict_dist = predict_dist
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.predict_dist:
            self.mlp_mu = build_mlps(
                in_channels, hidden_sizes, slot_size, use_bn=use_bn)
            self.mlp_log_sigma = build_mlps(
                in_channels, hidden_sizes, slot_size, use_bn=use_bn)
        else:
            self.mlp_mu = build_mlps(
                in_channels,
                hidden_sizes,
                num_slots * slot_size,
                use_bn=use_bn)

    def forward(self, text_features: Tensor):
        """Forward function.

        Args:
            text_features: [B, C], features extracted from sentences
        """
        slot_mu = self.mlp_mu(text_features)

        if self.predict_dist:
            slot_log_sigma = self.mlp_log_sigma(text_features)
            return slot_mu, slot_log_sigma

        return slot_mu.view(-1, self.num_slots, self.slot_size), None


class TransformerText2Slot(nn.Module):
    """Generate slot embedding from text features using TransformerDecoders.

    Args:
        in_channels (int): channels of input text features.
        d_model (int): hidden dims in Transformer
    """

    def __init__(self,
                 in_channels: int,
                 num_slots: int,
                 slot_size: int = 64,
                 d_model: int = 64,
                 nhead: int = 1,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(TransformerText2Slot, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        if in_channels != self.d_model:
            self.input_proj = nn.Linear(in_channels, self.d_model, bias=True)
        else:
            self.input_proj = nn.Identity()
        if slot_size != self.d_model:
            self.output_proj = nn.Linear(self.d_model, slot_size, bias=True)
        else:
            self.output_proj = nn.Identity()

        # Transformer decoder for query, language interaction
        decoder_layer = TransformerDecoderLayer(
            self.d_model,
            self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=activation)
        norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.num_layers, norm=norm)

        self._reset_parameters()

        # learnable queries to interact with language features
        self.query_embed = nn.Embedding(num_slots, slot_size)

    def forward(self, text_features: Tensor, text_padding_mask: Tensor = None):
        """Forward function.

        Args:
            text_features: [B, L, C], features extracted for *each* word.
            text_padding_mask: [B, L], mask indicating padded position
        """
        bs = text_features.shape[0]
        query_embed = self.query_embed.unsqueeze(0).repeat(bs, 1, 1)
        text_features = self.input_proj(text_features)
        pred_slots = self.decoder(
            query_embed.permute(1, 0, 2),
            text_features.permute(1, 0, 2),
            memory_key_padding_mask=text_padding_mask,
        ).permute(1, 0, 2)  # [B, num_slots, D]
        pred_slots = self.output_proj(pred_slots)
        return pred_slots, None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
