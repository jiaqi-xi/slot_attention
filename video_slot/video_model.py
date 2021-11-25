from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SlotAttentionModel


class RecurrentSlotAttentionModel(SlotAttentionModel):
    """Recurrent variant of original slot-attention model.

    Randomly initialize slot embedding at the beginning and do slot-att on the
        first frame, then use the resulting slot embedding as initilization for
        the following frames.

    Args:
        num_clips (int): Default clip number used in training. In testing, it's
            possible to input video of different lengths.
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        num_clips: int,
        num_slots: int,
        num_iterations: int = 2,
        kernel_size: int = 5,
        slot_size: int = 128,
        out_features: int = 64,
        enc_hiddens: Tuple[int, ...] = (3, 32, 32, 32, 32),
        use_unet: bool = False,
        dec_hiddens: Tuple[int, ...] = (128, 64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (8, 8),
        use_deconv: bool = True,
        slot_mlp_size: int = 256,
        learnable_slot: bool = True,
        stop_recur_slot_grad: bool = False,
        use_entropy_loss: bool = False,
    ):
        super().__init__(resolution, num_slots, num_iterations, kernel_size,
                         slot_size, out_features, enc_hiddens, use_unet,
                         dec_hiddens, decoder_resolution, use_deconv,
                         slot_mlp_size, learnable_slot, use_entropy_loss)

        self.num_clips = num_clips
        self.stop_recur_slot_grad = stop_recur_slot_grad

    def forward(self, x):
        """x: [B, num_clips, C, H, W]"""
        if self.training:
            return self._forward(x)
        # num_clips could be too long, manually do batch processing here
        B, num_clips, C, H, W = x.shape
        clip_len = num_clips  # every time feed this length of video to model
        while True:
            try:
                _ = self._forward(x[:, :clip_len])
                del _
                torch.cuda.empty_cache()
                break
            except RuntimeError:  # CUDA out of memory
                clip_len = clip_len // 2 + 1
        # no need to split
        if clip_len == num_clips:
            return self._forward(x)
        # batch processing
        all_rec_combs, all_recs, all_masks, all_slots = [], [], [], []
        prev_slots = None
        for clip_idx in range(0, num_clips, clip_len):
            rec_combs, recs, masks, slots = self._forward(
                x[:, clip_idx:clip_idx + clip_len], prev_slots)
            all_rec_combs.append(
                rec_combs.unflatten(0, (B, -1)).cpu().detach())
            all_recs.append(recs.unflatten(0, (B, -1)).cpu().detach())
            all_masks.append(masks.unflatten(0, (B, -1)).cpu().detach())
            all_slots.append(slots.unflatten(0, (B, -1)).cpu().detach())
            prev_slots = slots.unflatten(0, (B, -1))[:, -1].detach().clone()
            del rec_combs, recs, masks, slots
            torch.cuda.empty_cache()
        rec_combs = torch.cat(all_rec_combs, dim=1).flatten(0, 1)
        recs = torch.cat(all_recs, dim=1).flatten(0, 1)
        masks = torch.cat(all_masks, dim=1).flatten(0, 1)
        slots = torch.cat(all_slots, dim=1).flatten(0, 1)
        return rec_combs, recs, masks, slots

    def _forward(self, x, slot_prev=None):
        torch.cuda.empty_cache()

        num_clips = x.shape[1]
        x = x.flatten(0, 1)
        batch_size, num_channels, height, width = x.shape

        # TODO: for encoding, here we still use per-frame CNN
        # TODO: so we don't reshape a temporal dimension
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        # TODO: the core of Recurrent Slow Attention Model
        # reshape to [batch_size, num_clips, height*width, filter_size]
        batch_size = batch_size // num_clips
        encoder_out = encoder_out.reshape(batch_size, num_clips,
                                          height * width, -1)
        slots = []
        for clip_idx in range(num_clips):
            # [batch_size, self.num_slots, self.slot_size]
            one_slot = self.slot_attention(encoder_out[:, clip_idx], slot_prev)
            slots.append(one_slot)
            slot_prev = one_slot
            # optionally stop the grad of slot_prev
            if self.stop_recur_slot_grad:
                slot_prev = slot_prev.detach()
        # [batch_size*num_clips, self.num_slots, self.slot_size]
        slots = torch.stack(
            slots, dim=1).reshape(-1, self.num_slots, self.slot_size)

        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        # spatial broadcast
        decoder_in = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.decoder_resolution[0],
                                       self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, C, H, W]
        masks = out[:, :, -1:, :, :]  # [B, num_slots, 1, H, W]
        masks = F.softmax(masks, dim=1)
        # slot_recons = recons * masks + (1 - masks)
        recon_combined = torch.sum(recons * masks, dim=1)
        # recon_combined: [B, C, H, W]
        # recons: [B, num_slots, C, H, W]
        # masks: [B, num_slots, 1, H, W]
        # slots: [B, num_slots, slot_size]
        # slot_recons: [B, num_slots, C, H, W]
        # TODO: I return slot_recons instead of slots here!
        # TODO: this is different from the other slot-att models
        return recon_combined, recons, masks, slots  # slot_recons
