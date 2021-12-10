from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoModel

from clip import CLIP
from utils import Tensor, build_mlps


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
                 use_bn: bool = False):
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
                 text_length: int = 77,
                 d_model: int = 64,
                 nhead: int = 1,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 text_pe: bool = True,
                 out_mlp_layers: int = 2):
        super(TransformerText2Slot, self).__init__()

        # Transformer decoder for query, language interaction
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation)
        norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)
        # reset params as in MaskFormer
        self._reset_parameters()

        if in_channels != d_model:
            self.input_proj = nn.Linear(in_channels, d_model, bias=True)
        else:
            self.input_proj = nn.Identity()
        hidden_dims = [d_model for _ in range(out_mlp_layers - 1)]
        self.output_proj = build_mlps(
            d_model, hidden_dims, slot_size, use_bn=False)

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
        """Forward function.

        Args:
            text_features: [B, L, C], features extracted for *each* word.
            text_padding_mask: [B, L], mask indicating padded position
        """
        if isinstance(inputs, dict):
            text_features = inputs['text_features']
            text_padding_mask = inputs.get('text_padding_mask')
        else:
            text_features = inputs
            text_padding_mask = None
        bs = text_features.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        text_features = self.input_proj(text_features)
        if self.text_pe:  # do positional encoding
            text_features = text_features + \
                self.text_pos_embed.weight.unsqueeze(0)
        pred_slots = self.decoder(
            query_embed,
            text_features.permute(1, 0, 2).contiguous(),
            memory_key_padding_mask=text_padding_mask,
        ).permute(1, 0, 2).contiguous()  # [B, num_slots, D]
        pred_slots = self.output_proj(pred_slots)
        return pred_slots, None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class ObjMLPText2Slot(nn.Module):
    """Generate slot embedding from text features using MLPs.

    Input for each scene is [N, C]

    Args:
        in_channels (int): channels of input text features.
        hidden_sizes (Tuple[int]): MLPs hidden sizes.
        random_bg_slot (bool): Whether bg slot is learnable.
        bg_same_slot (bool): Whether input the same vector for bg slot.
    """

    def __init__(self,
                 in_channels: int,
                 slot_size: int,
                 hidden_sizes: Tuple[int] = (256, ),
                 use_bn: bool = False,
                 normalize_slots: bool = False):
        super(ObjMLPText2Slot, self).__init__()
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize_slots = normalize_slots

        # simple share-weight MLPs
        self.mlp = build_mlps(
            in_channels, hidden_sizes, slot_size, use_bn=use_bn)

    def forward(self, text_features: Tensor):
        """Forward function.

        Args:
            text_features: [B, num_slots, C], features extracted from sentences

        Returns:
            slots of shape [B, num_slots, slot_size]
        """
        assert len(text_features.shape) == 3
        slots = self.mlp(text_features)
        if self.normalize_slots:
            slots = F.normalize(slots, p=2, dim=-1)
        return slots, None  # None is for sigma


class CLIPTextEncoder(nn.Module):

    def __init__(self, clip_model: CLIP, context_len: int):
        super().__init__()

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final

        self.transformer_width = self.token_embedding.weight.shape[1]
        self.context_len = context_len
        if self.context_len > 0:
            # from CoOp paper, we random init a general context vector
            # and concat it in front of text
            self.context_embedding = nn.Parameter(
                nn.init.normal_(
                    torch.empty(
                        self.context_len,
                        self.transformer_width,
                        dtype=clip_model.dtype),
                    std=0.02))

    def _concat_context(self, text: Tensor, context: Tensor):
        x_text = self.token_embedding(text).type(self.dtype)  # [B, n_ctx, C]
        eos_idx = text.argmax(dim=-1)

        use_context = (context is not None) or (self.context_len > 0)
        if not use_context:
            return x_text, eos_idx

        B = x_text.shape[0]
        if context is None:
            context = self.context_embedding.unsqueeze(0).repeat(B, 1, 1)
        N2 = context.shape[1]

        # eos has the max token_id, add `N2` to get eos_idx after concat
        eos_idx = eos_idx + N2  # [B]
        assert eos_idx.max() < text.shape[1]
        x = torch.cat([x_text[:, :1], context, x_text[:, 1:-N2]], dim=1)
        return x, eos_idx

    def forward(self, text: Tensor, context: Tensor = None):
        """text: token_id of shape [B, n_ctx]"""
        x, eos_idx = self._concat_context(text, context)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [B, n_ctx, C]
        # take features from the eos embedding
        x = x[torch.arange(x.shape[0]), eos_idx]

        return x  # [B, C]

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp[0].weight.dtype


class TransformerTextEncoder(nn.Module):

    def __init__(self, model_name: str, context_len: int):
        super().__init__()

        print(f'Using {model_name} model from transformers lib')
        self.text_encoder = AutoModel.from_pretrained(model_name)

        self.transformer_width = \
            self.text_encoder.embeddings.word_embeddings.weight.shape[1]
        self.context_len = context_len
        if self.context_len > 0:
            # from CoOp paper, we random init a general context vector
            # and concat it in front of text
            self.context_embedding = nn.Parameter(
                nn.init.normal_(
                    torch.empty(
                        self.context_len,
                        self.transformer_width,
                        dtype=self.text_encoder.dtype),
                    std=0.02))

    def _concat_context(self, tokens: dict, context: Tensor):
        use_context = (context is not None) or (self.context_len > 0)
        if not use_context:
            return tokens

        input_ids = tokens['input_ids']  # [B, n_ctx] torch.long
        token_type_ids = tokens['token_type_ids']  # [B, n_ctx] all zeros
        attention_mask = tokens['attention_mask']  # [B, n_ctx] mask

        # in order to concat with context vectors, we need to manually get
        # the token embeddings out, shape [B, n_ctx, C]
        x_text = self.text_encoder.embeddings.word_embeddings(input_ids)
        B = x_text.shape[0]
        if context is None:
            context = self.context_embedding.unsqueeze(0).repeat(B, 1, 1)
        N2 = context.shape[1]
        x = torch.cat([x_text[:, :1], context, x_text[:, 1:]], dim=1)
        token_type_ids = torch.cat([
            token_type_ids[:, :1],
            torch.zeros((B, N2)).type_as(token_type_ids), token_type_ids[:, 1:]
        ],
                                   dim=1)
        attention_mask = torch.cat([
            attention_mask[:, :1],
            torch.ones((B, N2)).type_as(attention_mask), attention_mask[:, 1:]
        ],
                                   dim=1)
        tokens = {
            'inputs_embeds': x,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        return tokens

    def forward(self, tokens: dict, context: Tensor = None):
        tokens = self._concat_context(tokens, context)
        # take the [CLS] embedding as the sentence feature
        text_features = self.text_encoder(
            **tokens, return_dict=True)['last_hidden_state'][:, 0]
        return text_features
