import sys

sys.path.append('../')

import torch
from torch import nn
import torch.nn.functional as F

from model import SlotAttentionModel


class MoCoSlotAttentionModel(nn.Module):
    """MoCo wrapper for SlotAttentionModel.

    Args:
        model_q/k: base encoder, k is EMA of q
        K: queue size, i.e. the number of negative keys
        m: moco momentum of updating key encoder
        T: softmax temperature
    """

    def __init__(self,
                 model_q: SlotAttentionModel,
                 model_k: SlotAttentionModel,
                 dim: int = 64,
                 K: int = 4096,
                 m: float = 0.999,
                 T: float = 0.07):
        super().__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        self.model_q = model_q
        self.model_k = model_k.eval()

        for param_q, param_k in zip(self.model_q.parameters(),
                                    self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, data):
        """Forward function.

        Args:
            x (dict): Input data dict the the following keys:
                - img: [B, 2, C, H, W], each batch is same video's two frames
                - text: [B, 2, L], two frames in each batch share the same text
        """
        img, text = data['img'], data['text']
        img_q, img_k = img[:, 0], img[:, 1]
        text_q, text_k = text[:, 0], text[:, 1]
        x_q = dict(img=img_q, text=text_q)
        x_k = dict(img=img_k, text=text_k)
        recon_combined, recons, masks, slots_q = self.model_q(x_q)

        # if in testing, directly return the output of SlotAttentionModel
        if not self.training:
            return recon_combined, recons, masks, slots_q

        # compute query features
        # TODO: query features from Slot embeddings?
        # slots_q of shape [B, num_slots, dim] --> q of shape [N, dim]
        q = F.normalize(slots_q, dim=-1).view(-1, self.dim)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # TODO: no shuffling since we're not in DDP training
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            slots_k = self.model_k.encode(x_k)  # keys: [B, num_slots, dim]
            k = F.normalize(slots_k, dim=-1).view(-1, self.dim)  # [N, dim]

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        logits, labels = self._moco_infonce(q, k)

        return recon_combined, recons, masks, slots_q, logits, labels

    def loss_function(self, input):
        """Calculate reconstruction loss and contrastive loss."""
        if not self.training:
            recon_combined, _, masks, _ = self.forward(input)
            recon_loss = F.mse_loss(recon_combined, input['img'][:, 0])
            loss_dict = {
                'recon_loss': recon_loss,
            }
        else:
            recon_combined, _, masks, _, logits, labels = self.forward(input)
            recon_loss = F.mse_loss(recon_combined, input['img'][:,
                                                                 0])  # img_q
            contrastive_loss = F.cross_entropy(logits, labels)
            loss_dict = {
                'recon_loss': recon_loss,
                'contrastive_loss': contrastive_loss,
            }

        # masks: [B, num_slots, 1, H, W], apply entropy loss
        if self.model_q.use_entropy_loss:
            masks = masks[:, :, 0]  # [B, num_slots, H, W]
            entropy_loss = (-masks * torch.log(masks + 1e-6)).sum(1).mean()
            loss_dict['entropy'] = entropy_loss
        return loss_dict

    def _moco_infonce(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: [N, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: [N, (1+K)]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    @torch.no_grad()
    def _momentum_update_model_k(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(),
                                    self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # TODO: only used in dist training
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
