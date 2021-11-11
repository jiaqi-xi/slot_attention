import sys
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('../')

from model import SlotAttentionModel
import numpy as np
import lpips


class PerceptualLoss(nn.Module):

    def __init__(self, arch='vgg'):
        super().__init__()

        assert arch in ['alex', 'vgg', 'squeeze']
        self.loss_fn = lpips.LPIPS(net=arch).eval()
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    def loss_function(self, x_prev, x_future):
        """x_prev and x_future are of same shape.
        Should be mask * recon + (1 - mask)
        """
        assert len(x_prev.shape) == len(x_future.shape) == 4
        x_prev = torch.clamp(x_prev, min=-1., max=1.)
        x_future = torch.clamp(x_future, min=-1., max=1.)
        loss = self.loss_fn(x_prev, x_future).mean()
        return loss


class PerceptualSlotAttentionModel(nn.Module):
    """SlotAttentionModel that uses Perceptual learning loss.

    Args:
        model: base encoder, k is EMA of q
        T: softmax temperature
        mlp: additional projection head from slot emb to contrastive features
    """

    def __init__(self,
                 model: SlotAttentionModel,
                 dim: int = 64):
        super().__init__()

        self.dim = dim
        self.T = T

        # projection head
        self.mlp = (mlp is not None and len(mlp) > 0)
        if self.mlp:
            assert isinstance(mlp, (list, tuple))
            assert mlp[-1] == self.dim
            model.proj_head = build_mlps(
                model.slot_size, mlp[:-1], mlp[-1], use_bn=False)
            print('Using MLP projection head in perceptual learning!')

        self.model = model
        self.perceptual_loss = PerceptualLoss()
        self.num_slots = self.model.num_slots

    def forward_test(self, data):
        return self.model(dict(img=data['img'], text=data['text']))

    def forward(self, data):
        """Forward function.

        Args:
            x (dict): Input data dict the the following keys:
                - img: [B, C, H, W], image as q
                - text: [B, L], text corresponding to img
                - img2: [B, C, H, W], img as k
                - text2: [B, L], text corresponding to img2
        """
        # if in testing, directly return the output of SlotAttentionModel
        if not self.training:
            return self.forward_test(data)

        img = torch.cat([data['img'], data['img2']], dim=0)
        text = torch.cat([data['text'], data['text2']], dim=0)
        x = dict(img=img, text=text)
        recon_combined, recons, masks, slots = self.model(x)

        return recon_combined, recons, masks, slots, None, None

    def _build_n_pair(self, feats):
        """Construct anchor, positive and negatives for N-pair loss"""
        # [2, B, num_slots, dim]
        feats = feats.view(2, -1, self.num_slots, self.dim)

        # `anchors` is of shape [B * num_slots, dim]
        anchors = feats[0].view(-1, self.dim)
        # `positives` is of shape [B * num_slots, dim]
        positives = feats[1].view(-1, self.dim)
        # `negatives` is of shape [B * num_slots, 2*(num_slots-1), dim]
        negatives = []
        for slot_idx in range(self.num_slots):
            negatives.append(  # each is of shape [B, 2*(num_slots-1), dim]
                torch.cat([feats[:, :, :slot_idx], feats[:, :, slot_idx + 1:]],
                          dim=2).transpose(0, 1).flatten(1, 2))
        negatives = torch.stack(negatives, dim=1).flatten(0, 1)
        return anchors, positives, negatives

    def loss_function(self, input):
        """Calculate reconstruction loss and contrastive loss."""
        if not self.training:
            recon_combined, _, masks, _ = self.forward(input)
            recon_loss = F.mse_loss(recon_combined, input['img'])
            loss_dict = {
                'recon_loss': recon_loss,
            }
        else:
            recon_combined, recons, masks, _, _, _ = self.forward(input)
            img = torch.cat([input['img'], input['img2']], dim=0)
            recon_loss = F.mse_loss(recon_combined, img)
            recon_1, recon_2 = np.split(recons, 2, axis=0)
            mask_1, mask_2 = np.split(masks, 2, axis=0)
            x_1 = mask_1 * recon_1 + (1 - mask_1)
            x_2 = mask_2 * recon_2 + (1 - mask_2)
            perceptual_loss = self.perceptual_loss.loss_function(x_1, x_2)
            loss_dict = {
                'recon_loss': recon_loss,
                'perceptual_loss': perceptual_loss,
            }

        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.model.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy'] = entropy_loss
        return loss_dict


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
