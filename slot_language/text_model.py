from typing import Tuple

import torch
from torch import nn

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


class Text2Slot(nn.Module):
    """Generate slot embedding from text features.

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
        super(Text2Slot, self).__init__()
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
