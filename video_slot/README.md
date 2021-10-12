# Apply slot attention to Video

The code in this folder is mainly based on `clevr_video/`. Instead of applying slot-attention to every individual video frame, we aim at improve the temporal continuity of slots, i.e. make one slot "tracking" one object through time.
Currently I modify the dataloader so that every time we will load a clip of video of shape `[B, num_clip, C, H, W]` instead of one frame `[B, C, H, W]`, which is controlled by the hyper-parameter `sample_clip_num` in `params.py`.

In order to enforce temporal consistency, I've implemented several possible solutions:
- Future prediction. Build a ConvAE like network to predict `mask_{t+1}` given `mask_t`, or predict `slot-recon_{t+1}` given `slot_recon_t`. However, it seems that it doesn't work well.
- Perceptual loss. One slot captures one object means that `slot-recon_{t}` should be (semantically) similar to `slot-recon_{t+1}`. So I use a [PerceptualLoss](https://github.com/richzhang/PerceptualSimilarity) to enforce similarity between slots. By setting `perceptual_loss='alex'` we can turn on this loss. Haven't been able to run the experiment yet...


TBH I don't really think future prediction or perceptual loss can really solve the problem. Should try something like video feature extraction network (e.g. S3D), or ConvLSTM in the future.
A most naive idea is to initialize `slot_{t+1}` with the slot feature at time `t`. By doing so, it's similar to that we regard `slot` as the hidden_state in a LSTM, the `slot-recon` and `mask` as the output of LSTM. Will try that out in the future.
