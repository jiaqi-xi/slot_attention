import sys
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('../')

from model import SlotAttentionModel


class MoCoSlotAttentionModel(nn.Module):
    """MoCo wrapper for SlotAttentionModel.

    Args:
        model_q/k: base encoder, k is EMA of q
        K: queue size, i.e. the number of negative keys
        m: moco momentum of updating key encoder
        T: softmax temperature
        mlp: additional projection head from slot emb to contrastive features
        diff_video: whether to avoid same video k in contrastive loss
    """

    def __init__(self,
                 model_q: SlotAttentionModel,
                 model_k: SlotAttentionModel,
                 dim: int = 64,
                 K: int = 8192,
                 m: float = 0.999,
                 T: float = 0.07,
                 mlp: Optional[int] = None,
                 diff_video: bool = False):
        super().__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.diff_video = diff_video

        # TODO: projection head
        self.mlp = (mlp is not None)
        if self.mlp:
            assert isinstance(mlp, (list, tuple))
            assert mlp[-1] == self.dim
            model_q.proj_head = build_mlps(
                model_q.slot_size, mlp[:-1], mlp[-1], use_bn=False)
            model_k.proj_head = build_mlps(
                model_k.slot_size, mlp[:-1], mlp[-1], use_bn=False)
            print('Using MLP projection head in contrastive learning!')

        self.model_q = model_q
        self.model_k = model_k.eval()
        self.num_slots = model_q.num_slots

        for param_q, param_k in zip(self.model_q.parameters(),
                                    self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # in case `diff_video=True`, every time we sample a subset of queue
        # to keep the same number of negatives within a batch
        # since queue_size < (num_video * clip_len) (total clip num in dataset)
        # max_sample_num == (queue_size - clip_len)
        self.register_buffer("queue_vid_id", -torch.ones(K).long())
        if self.diff_video:
            self.sample_num = self.K - 64  # CLEVR clip_len == 34
        else:
            self.sample_num = self.K

    def forward(self, data):
        """Forward function.

        Args:
            x (dict): Input data dict the the following keys:
                - img: [B, C, H, W], image as q
                - text: [B, L], text corresponding to img
                - img2: [B, C, H, W], img as k
                - text2: [B, L], text corresponding to img2
                - video_idx: [B]
        """
        img_q, text_q = data['img'], data['text']
        x_q = dict(img=img_q, text=text_q)
        recon_combined, recons, masks, slots_q = self.model_q(x_q)

        # if in testing, directly return the output of SlotAttentionModel
        if not self.training:
            return recon_combined, recons, masks, slots_q

        # compute query features
        # TODO: query features from Slot embeddings?
        # slots_q of shape [B, num_slots, dim] --> q of shape [N, dim]
        q = slots_q.view(-1, self.dim)
        if self.mlp:
            q = self.model_q.proj_head(q)
        q = F.normalize(q, dim=1)  # [N, dim]

        # compute key features
        img_k, text_k = data['img2'], data['text2']
        x_k = dict(img=img_k, text=text_k)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_model_k()  # update the key encoder

            # shuffle for making use of BN
            # TODO: no shuffling since we're not in DDP training
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            slots_k = self.model_k.encode(x_k)  # keys: [B, num_slots, dim]
            k = slots_k.view(-1, self.dim)
            if self.mlp:
                k = self.model_k.proj_head(k)
            k = F.normalize(k, dim=1)  # [N, dim]

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        logits, labels = self._run_moco(q, k, data.get('video_idx'))

        return recon_combined, recons, masks, slots_q, logits, labels

    def loss_function(self, input):
        """Calculate reconstruction loss and contrastive loss."""
        if not self.training:
            recon_combined, _, masks, _ = self.forward(input)
            recon_loss = F.mse_loss(recon_combined, input['img'])
            loss_dict = {
                'recon_loss': recon_loss,
            }
        else:
            recon_combined, _, masks, _, logits, labels = self.forward(input)
            recon_loss = F.mse_loss(recon_combined, input['img'])  # img_q
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

    def _run_moco(self, q, k, vid_ids=None):
        """Run MoCo style InfoNCE loss.

        Args:
            q/k: [B * num_slots (N), dim], positive pairs
            vid_ids: if not None, should be shape [B]
        """
        if not self.diff_video:
            vid_ids = None  # no need to consider
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: [N, K]
        if vid_ids is None:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        else:
            negatives = self._sample_negative(vid_ids)
            assert len(negatives.shape) == 3  # [dim, B, sample_num]
            negatives = negatives.unsqueeze(2).repeat(  # [dim, N, sample_num]
                1, 1, self.num_slots, 1).flatten(start_dim=1, end_dim=2)
            l_neg = torch.einsum('nc,cnm->nm', [q, negatives])

        # logits: [N, (1+K)]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, vid_ids)

        return logits, labels

    @torch.no_grad()
    def _sample_negative(self, vid_ids=None):
        # vid_ids is of shape [B]
        # self.queue is of shape [dim, K]
        if vid_ids is None or not self.diff_video:
            assert self.sample_num == self.K
            return self.queue
        # get i's negative samples whose vid_id != vid_ids[i]
        all_idx = torch.arange(self.K).long().to(vid_ids.device)
        sample_idx = torch.stack([
            all_idx[self.queue_vid_id != vid_ids[i]][:self.sample_num]
            for i in range(vid_ids.shape[0])
        ],
                                 dim=0)  # [B, self.sample_num]
        negatives = self.queue[:, sample_idx]  # [dim, B, self.sample_num]
        return negatives.clone().detach()

    @torch.no_grad()
    def _momentum_update_model_k(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(),
                                    self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, vid_ids=None):
        # gather keys before updating queue
        # TODO: only used in dist training
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if vid_ids is not None:
            self.queue_vid_id[ptr:ptr + batch_size] = vid_ids
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
