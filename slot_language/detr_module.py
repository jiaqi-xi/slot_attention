# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DETRText2Slot(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_slots: int,
        slot_size: int = 64,
        text_length: int = 77,
        d_model: int = 64,
        nhead: int = 1,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = 'relu',
        text_pe: bool = True,
        out_mlp_layers: int = 2,
    ):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers,
            decoder_norm,
        )

        self._reset_parameters()

        if in_channels != d_model:
            self.input_proj = nn.Linear(in_channels, d_model, bias=True)
        else:
            self.input_proj = nn.Identity()
        self.output_proj = MLP(d_model, d_model, slot_size, out_mlp_layers)

        # learnable queries to interact with language features
        self.query_embed = nn.Embedding(num_slots, d_model)
        nn.init.xavier_uniform_(  # as the slot_mu/sigma in slot-attention
            self.query_embed.weight,
            gain=nn.init.calculate_gain("linear"))

        # learnable positional embedding for text features
        self.text_pe = text_pe
        if self.text_pe:
            self.text_pos_embed = nn.Embedding(text_length, d_model)
            nn.init.normal_(self.text_pos_embed.weight, std=0.01)

    def forward(self, inputs: dict):
        if isinstance(inputs, dict):
            text_features = inputs['text_features']
            text_padding_mask = inputs.get('text_padding_mask')
        else:
            text_features = inputs
            text_padding_mask = None
        bs = text_features.shape[0]
        text_features = self.input_proj(text_features)

        pos_embed = self.text_pos_embed.weight.unsqueeze(1).repeat(
            1, bs, 1) if self.text_pe else None
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        pred_slots = self.decoder(
            tgt,
            text_features.permute(1, 0, 2).contiguous(),
            memory_key_padding_mask=text_padding_mask,
            pos=pos_embed,
            query_pos=query_embed).permute(1, 0, 2).contiguous()

        pred_slots = self.output_proj(pred_slots)
        return pred_slots, None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
