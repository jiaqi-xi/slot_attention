import torch
import torch.nn as nn


def get_normalizer(norm, channels):
    assert norm in ['', 'bn', 'gn', 'in']
    if norm == '':
        normalizer = nn.Identity()
    elif norm == 'bn':
        normalizer = nn.BatchNorm2d(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        normalizer = nn.GroupNorm(channels // 16, channels)
    elif norm == 'in':
        normalizer = nn.InstanceNorm2d(channels)
    return normalizer


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


def build_mlps(in_channels, hidden_sizes, out_channels, use_bn=False):
    if hidden_sizes is None or len(hidden_sizes) == 0:
        return nn.Linear(in_channels, out_channels)
    modules = [fc_bn_relu(in_channels, hidden_sizes[0], use_bn=use_bn)]
    for i in range(0, len(hidden_sizes) - 1):
        modules.append(
            fc_bn_relu(hidden_sizes[i], hidden_sizes[i + 1], use_bn=use_bn))
    modules.append(nn.Linear(hidden_sizes[-1], out_channels))
    return nn.Sequential(*modules)


class SepLinear(nn.Module):
    """Apply two nn.Linear to two sub-part of the input tensors separately."""

    def __init__(self, sep_idx, in_dim, out_dim, bias=True):
        super().__init__()

        assert isinstance(sep_idx, int)
        sep_ratio = sep_idx / in_dim
        out_dim1 = int(out_dim * sep_ratio)
        self.sep_idx = sep_idx
        self.linear1 = nn.Linear(sep_idx, out_dim1, bias)
        self.linear2 = nn.Linear(in_dim - sep_idx, out_dim - out_dim1, bias)

    def forward(self, x):
        return torch.cat([
            self.linear1(x[..., :self.sep_idx]),
            self.linear2(x[..., self.sep_idx:])
        ],
                         dim=-1)


class SepLayerNorm(nn.Module):

    def __init__(self, sep_idx, norm_dim, eps=1e-5, elementwise_affine=True):
        super().__init__()

        assert isinstance(norm_dim, int) and isinstance(sep_idx, int)
        self.sep_idx = sep_idx
        self.ln1 = nn.LayerNorm(sep_idx, eps, elementwise_affine)
        self.ln2 = nn.LayerNorm(norm_dim - sep_idx, eps, elementwise_affine)

    def forward(self, x):
        return torch.cat(
            [self.ln1(x[..., :self.sep_idx]),
             self.ln2(x[..., self.sep_idx:])],
            dim=-1)


class SepGRUCell(nn.Module):

    def __init__(self, sep_idx, input_size, hidden_size, bias=True):
        super().__init__()

        assert isinstance(sep_idx, int)
        sep_ratio = sep_idx / input_size
        hidden_size1 = int(hidden_size * sep_ratio)
        self.sep_idx = sep_idx
        self.gru1 = nn.GRUCell(sep_idx, hidden_size1, bias)
        self.gru2 = nn.GRUCell(input_size - sep_idx,
                               hidden_size - hidden_size1, bias)

    def forward(self, input, hx=None):
        if hx is None:
            hx1, hx2 = None, None
        else:
            hx1, hx2 = hx[..., :self.sep_idx], hx[..., self.sep_idx:]
        return torch.cat([
            self.gru1(input[..., :self.sep_idx], hx1),
            self.gru2(input[..., self.sep_idx:], hx2)
        ],
                         dim=-1)
