from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from obj_model import ObjSlotAttentionModel
from obj_utils import build_mlps


class ObjAugSlotAttentionModel(nn.Module):

    def __init__(
            self,
            model: ObjSlotAttentionModel,
            eps: float = 1e-6,
            contrastive_loss_dict=dict(
                use_contrastive_loss=False,
                contrastive_mlp=(),
                contrastive_T=0.1,
                contrastive_normalize=True,
                contrastive_stop_grad=False,
            ),
            text_recon_loss_dict=dict(
                use_text_recon_loss=False,
                text_recon_mlp=(64, ),
                text_recon_normalize=False,
            ),
            feature_loss_dict=dict(use_feature_loss=False, ),
    ):
        super().__init__()

        self.model = model
        self.eps = eps

        # contrastive loss
        self.use_contrastive_loss = contrastive_loss_dict.use_contrastive_loss
        self.contrastive_T = contrastive_loss_dict.contrastive_T
        self.contrastive_normalize = contrastive_loss_dict.contrastive_normalize
        self.contrastive_stop_grad = contrastive_loss_dict.contrastive_stop_grad
        self.num_slots = self.model.num_slots
        self.slot_size = self.model.slot_size
        # index used for contrastive loss computation
        if self.use_contrastive_loss:
            sample_num = 2 * self.num_slots
            neg_idx = [list(range(sample_num)) for _ in range(sample_num)]
            for idx in range(sample_num):  # remove self and pos pair
                neg_idx[idx].remove(idx)
                neg_idx[idx].remove((idx + self.num_slots) % sample_num)
            self.register_buffer('neg_idx', torch.tensor(neg_idx).long())
            contrastive_mlp = contrastive_loss_dict.contrastive_mlp
            if len(contrastive_mlp) >= 1:
                self.contrastive_mlp = build_mlps(self.slot_size,
                                                  contrastive_mlp[:-1],
                                                  contrastive_mlp[-1], False)
            else:
                self.contrastive_mlp = nn.Identity()

        # text reconstruction loss
        self.use_text_recon_loss = text_recon_loss_dict.use_text_recon_loss
        self.text_recon_normalize = text_recon_loss_dict.text_recon_normalize
        self.text_feats_size = self.model.text2slot_model.in_channels
        if self.use_text_recon_loss:
            self.text_recon_mlp = build_mlps(
                self.slot_size, text_recon_loss_dict.text_recon_mlp,
                self.text_feats_size, False)

        # feature loss
        self.use_feature_loss = feature_loss_dict.use_feature_loss

    def forward_test(self, data):
        return self.model(dict(img=data['img'], text=data['text']))

    def forward(self, data):
        """Forward function.

        Args:
            data (dict): Input data dict containing the following items:
                - img/flipped_img: One frame and its (potentially) flipped version
                - is_flipped: Boolean
                - text: Text description corresponding to img
                - shuffled_text: Shuffled text, used for `flip_img`
                - shuffled_idx: Order of the shuffled text, [B, num_slots]
                - is_shuffled: Boolean
        """
        if not self.training:
            return self.forward_test(data)

        # at least one augmentation is applied
        assert data['is_flipped'][0].item() or data['is_shuffled'][0].item()
        x = dict(
            img=torch.cat([data['img'], data['flipped_img']], dim=0),
            text=torch.cat([data['text'], data['shuffled_text']], dim=0))
        recon_combined, recons, masks, slots, \
            img_feats, text_feats = self.model(x)

        return recon_combined, recons, masks, slots, img_feats, text_feats

    def loss_function(self, input):
        """Calculate loss.

        Three loss components:
            - MSE reconstruction loss
            - Equivariance loss
            - Entropy loss (optional)
            - Contrastive loss (optional)
        """
        if not self.training:
            return self.model.eval_loss_function(input)

        # recon_combined: [2 * B, 3, H, W]
        # recons: [2 * B, num_slots, 3, H, W]
        # masks: [2 * B, num_slots, 1, H, W]
        # slots: [2 * B, num_slots, slot_size]
        # img_feats: [2 * B, C_img, h, w]
        # text_feats: [2 * B, num_slots, C_text]
        recon_combined, recons, masks, slots, \
            img_feats, text_feats = self.forward(input)
        loss_dict = self.model.calc_train_loss(
            torch.cat([input['img'], input['flipped_img']], dim=0),
            recon_combined, recons, masks, slots, img_feats, text_feats)

        is_flipped = input['is_flipped'][0].item()
        is_shuffled = input['is_shuffled'][0].item()
        shuffled_idx = input['shuffled_idx'].long()  # [B, num_slots]

        loss_dict['equivariance_loss'] = self.calc_equivariance_loss(
            masks, is_flipped, is_shuffled, shuffled_idx)
        if self.use_contrastive_loss:
            loss_dict['contrastive_loss'] = self.calc_contrastive_loss(
                slots, is_shuffled, shuffled_idx)
        if self.use_text_recon_loss:
            loss_dict['text_recon_loss'] = self.calc_text_recon_loss(
                slots, text_feats)
        if self.use_feature_loss:
            loss_dict['feature_loss'] = self.calc_feature_loss(
                masks[:, :, 0], img_feats)
        return loss_dict

    def calc_equivariance_loss(self, masks, is_flipped, is_shuffled,
                               shuffled_idx):
        """Transformation equivariance loss."""
        bs = shuffled_idx.shape[0]
        masks = masks[:, :, 0]  # [2*B, num_slots, H, W]
        masks = masks + self.eps
        masks = masks / masks.sum(dim=1, keepdim=True)
        masks1, masks2 = masks[:bs], masks[bs:]  # [B, num_slots, H, W]
        if is_flipped:
            masks2 = TF.hflip(masks2)
        if is_shuffled:
            masks1 = masks1[torch.arange(bs)[:, None], shuffled_idx]

        # masks are probability tensors, however torch.kld requires log-prob
        equivariance_loss = F.kl_div(
            torch.log(masks1), masks2, reduction='mean') + F.kl_div(
                torch.log(masks2), masks1, reduction='mean')
        return equivariance_loss

    def calc_contrastive_loss(self, slots, is_shuffled, shuffled_idx):
        """Contrasting slots between different id."""
        bs = shuffled_idx.shape[0]
        slots = self.contrastive_mlp(slots)
        slots1, slots2 = slots[:bs], slots[bs:]  # [B, num_slots, slot_size]
        if is_shuffled:
            slots1 = slots1[torch.arange(bs)[:, None], shuffled_idx]
        # like in MoCo, maybe this can increase stability?
        if self.contrastive_stop_grad:
            slots2 = slots2.detach()
        slots = torch.cat([slots1, slots2], dim=1)
        # slots shape [B, num_slots * 2, slot_size]

        # construct anchor, positive, negative pairs
        # anchor and pos is [B * num_slots * 2, slot_size]
        # neg is [B * num_slots * 2, num_neg, slot_size]
        anchor = torch.cat([slots1, slots2], dim=1).flatten(0, 1)
        pos = torch.cat([slots2, slots1], dim=1).flatten(0, 1)
        neg = slots[torch.arange(bs)[:, None, None],
                    self.neg_idx[None]].flatten(0, 1)
        if self.contrastive_normalize:
            anchor = F.normalize(anchor, p=2, dim=-1)
            pos = F.normalize(pos, p=2, dim=-1)
            neg = F.normalize(neg, p=2, dim=-1)

        # pos: [N, 1], neg: [N, K]
        # CE loss by setting the first label as GT
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nmc->nm', [anchor, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.contrastive_T
        labels = torch.zeros(logits.shape[0]).long().to(logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        return contrastive_loss

    def calc_text_recon_loss(self, slots, text_feats):
        """Reconstruction loss from slot embedding to text features."""
        if self.text_recon_normalize:
            text_feats = F.normalize(text_feats, p=2, dim=-1)
        pred_text_feats = self.text_recon_mlp(slots)  # [2*B, num_slots, C]
        text_recon_loss = F.mse_loss(pred_text_feats, text_feats)
        return text_recon_loss

    def calc_feature_loss(self, masks, feats):
        # masks: N x R x H x W, probability mask after softmax
        # feats: N x C x H x W
        # N x R x C x H x W
        weighted_regions = masks.unsqueeze(2) * feats.unsqueeze(1)
        # N x R x 1
        mask_sum = masks.sum(3).sum(2, keepdim=True)
        # N x R x C
        means = weighted_regions.sum(4).sum(3) / mask_sum
        # N x R x C x H x W
        diff_sq = (feats.unsqueeze(1) - means.unsqueeze(3).unsqueeze(4))**2
        # N x R x C
        loss = (diff_sq * masks.unsqueeze(2)).mean(4).mean(3)
        return loss.sum(2).sum(1).mean()


class SemPosSepObjAugSlotAttentionModel(ObjAugSlotAttentionModel):

    def loss_function(self, input):
        """Calculate loss.

        Three loss components:
            - MSE reconstruction loss
            - Equivariance loss
            - Entropy loss (optional)
            - Contrastive loss (optional)
        """
        if not self.training:
            return self.model.eval_loss_function(input)

        recon_combined, recons, masks, slots, \
            img_feats, text_feats = self.forward(input)
        slots, sem_slots, pos_slots = slots
        loss_dict = self.model.calc_train_loss(
            torch.cat([input['img'], input['flipped_img']], dim=0),
            recon_combined, recons, masks, slots, img_feats, text_feats)

        is_flipped = input['is_flipped'][0].item()
        is_shuffled = input['is_shuffled'][0].item()
        shuffled_idx = input['shuffled_idx'].long()  # [B, num_slots]

        loss_dict['equivariance_loss'] = self.calc_equivariance_loss(
            masks, is_flipped, is_shuffled, shuffled_idx)
        if self.use_contrastive_loss:
            # we only supervise on the semantic slot embedding
            loss_dict['contrastive_loss'] = self.calc_contrastive_loss(
                sem_slots, is_shuffled, shuffled_idx)
        if self.use_text_recon_loss:
            loss_dict['text_recon_loss'] = self.calc_text_recon_loss(
                sem_slots, text_feats)
        if self.use_feature_loss:
            loss_dict['feature_loss'] = self.calc_feature_loss(
                masks[:, :, 0], img_feats)
        return loss_dict
