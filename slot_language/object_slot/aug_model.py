import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from obj_model import ObjSlotAttentionModel


class ObjAugSlotAttentionModel(nn.Module):

    def __init__(self, model: ObjSlotAttentionModel):
        super().__init__()

        self.model = model

    def forward_test(self, data):
        return self.model(
            dict(img=data['img'], text=data['text'], padding=data['padding']))

    def forward(self, data):
        """Forward function.

        Args:
            data (dict): Input data dict containing the following items:
                - img/flipped_img: One frame and its (potentially) flipped version
                - is_flipped: Boolean
                - text: Text description corresponding to img
                - padding: Pad 0 for background slots, [B, num_slots]
                - shuffled_text: Shuffled text, used for `flip_img`
                - shuffled_idx: Order of the shuffled text, [B, num_slots]
                - is_shuffled: Boolean
        """
        if not self.training:
            return self.forward_test(data)

        assert data['is_flipped'] or data['is_shuffled']
        x = dict(
            img=torch.cat([data['img'], data['flipped_img']], dim=0),
            text=torch.cat([data['text'], data['shuffled_text']], dim=0),
            padding=data['padding'].repeat(2, 1))
        recon_combined, recons, masks, slots = self.model(x)

        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        """Calculate loss.

        Three loss components:
            - MSE reconstruction loss
            - Equivariance loss
            - Entropy loss
        """
        recon_combined, recons, masks, slots = self.forward(input)
        if not self.training:
            recon_loss = F.mse_loss(recon_combined, input['img'])
            return {
                "recon_loss": recon_loss,
            }
        recon_loss = F.mse_loss(
            recon_combined,
            torch.cat([input['img'], input['flipped_img']], dim=0))
        loss_dict = {
            "recon_loss": recon_loss,
        }
        masks = masks[:, :, 0]  # [2*B, num_slots, H, W]
        if self.model.use_entropy_loss:
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy'] = entropy_loss

        # Equivariance loss
        padding = input['padding']  # [B, num_slots]
        bs = padding.shape[0]
        obj_mask = (padding == 1)
        masks1, masks2 = masks[:bs], masks[bs:]  # [B, num_slots, H, W]
        if input['is_flipped']:
            masks2 = TF.hflip(masks2)
        if not input['is_shuffled']:  # we only penalize foreground obj masks
            masks1, masks2 = masks1[obj_mask], masks2[obj_mask]  # [M, H, W]
        else:
            shuffled_idx = input['shuffled_idx'].long()  # [B, num_slots]
            masks1 = torch.cat([
                masks1[i, obj_mask[i]][shuffled_idx[i, obj_mask[i]]]
                for i in range(bs)
            ],
                               dim=0)
            masks2 = masks2[obj_mask]
        # masks are probability tensors, however torch.kld requires log-prob
        eps = 1e-6
        equivariance_loss = F.kl_div(torch.log(masks1 + eps), masks2) + \
            F.kl_div(torch.log(masks2 + eps), masks1)
        loss_dict['equivariance_loss'] = equivariance_loss
        return loss_dict
