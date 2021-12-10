import torch.nn as nn

from clip import CLIP


class CLIPVisionEncoder(nn.Module):

    def __init__(self, clip_model: CLIP):
        super().__init__()

        self.visual = clip_model.visual
        self.vit = clip_model.vit

    def forward(self, image):
        in_type = image.dtype
        if self.vit:
            # ViT
            # [B, N**2, C] (N == height // patch_size)
            feats = self.visual(
                image.type(self.dtype), global_feats=False, lin_proj=False)
            B, N2, C = feats.shape
            N = int(N2**0.5)
            # return of shape [B, C, N, N], mimicing a normal feature map
            feats = feats.reshape(B, N, N, C).permute(0, 3, 1, 2)
        else:
            # ResNet
            # [B, H*W, C]
            feats, (H, W) = self.visual(
                image.type(self.dtype), global_feats=False, no_pool=False)
            B, _, C = feats.shape
            # return a low resolution feature map
            feats = feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return feats.type(in_type)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
