import torch
import torch.nn as nn

from clip import CLIP


class CLIPVisionEncoder(nn.Module):

    def __init__(self, clip_model: CLIP):
        super().__init__()

        self.visual = clip_model.visual
        self.vit = clip_model.vit

    def forward(self, image, lin_proj=False, res_no_pool=False):
        in_type = image.dtype
        if self.vit:
            # ViT
            # [B, N**2, C] (N == height // patch_size) or [B, C]
            feats = self.visual(
                image.type(self.dtype), global_feats=False, lin_proj=lin_proj)
            B, N2, C = feats.shape
            N = int(N2**0.5)
            # return of shape [B, C, N, N], mimicing a normal feature map
            feats = feats.reshape(B, N, N, C).permute(0, 3, 1, 2)
        else:
            # ResNet
            # [B, H*W, C]
            feats, (H, W) = self.visual(
                image.type(self.dtype),
                global_feats=False,
                no_pool=res_no_pool)
            B, _, C = feats.shape
            # return a low resolution feature map
            feats = feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return feats.type(in_type)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


class CLIPTextEncoder(nn.Module):

    def __init__(self, clip_model: CLIP):
        super().__init__()

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, text, lin_proj=True):
        """If features are used for similarity calculation, we need `lin_proj`.
        text: token_id of shape [B, n_ctx]
        """
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        if lin_proj:
            # project to embedding space for contrastive learning
            x = x @ self.text_projection

        return x  # [B, C]

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp[0].weight.dtype
