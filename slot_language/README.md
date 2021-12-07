# Language Guided Slot Attention

The difference with former slot attention:

-   Replace the simple vision encoder with ViT from pre-trained CLIP
-   Introduce text as model input, encode text feature with the language branch from pre-trained CLIP. The text is something like `put the red cylinder to the left of the cyan cube`, which include action and the objects involved in it
-   Instead of learning a `mu` and `sigma` for each slot, we train a model to generate slot initialization from the text feature

So the biggest problem here is that, what is the supervision signal we have? We only have the reconstruction loss now, but we need semantic constraint, that's our desired model output!

Things that can be tuned now:

-   SlotModel:
    -   Vision encoder:
        -   `clip_arch`: default is `ViT`, I think that's fine and we can try other variants like ResNet50 if necessary, _could try_
        -   `clip_global_feats`: by default we take the features of each patch, since I believe the global one is trained in contrastive learning, while our task somehow requires spatial information to localize objects, **we should try**
    -   Slot attention module:
        -   `num_slots`: default is 7, that counts for `max_obj_num + 1`. But we have the prior here that, at most 2 objects are involved in one action, can we inject inductive bias via setting less slots here? **we should try**
        -   `slot_size`: default is 64, IMO that's fine, maybe try enlarging it to 128, _could try_
        -   `num_iterations`: default is 3, seems to be okay from the paper, _no need to try_
        -   `slot_mlp_size`: default is 128, so this is FFN's hidden size, _could try_
    -   Decoder module: the overall architecture is a big problem, it might be enough for this task but definitely not for real-world videos
        -   `dec_resolution`: default is (7, 7), which means we need 5 up-sampling, (14, 14) is another option. **we should try**
        -   `dec_kernel_size`: default is 3 which is suggested by `Object Radiance Field` paper, should try 5 as original paper. **we should try**
        -   `dec_channels`: default is 5 consecutive 64 channels, may adjust with `slot_size`, of course up=sampling+Conv could replace stride2 DeConv. **we should try**
-   Text2Slot module:
    -   `text2slot_hidden_sizes`: default is 2layer MLP with 256 as hidden size, maybe one FC is enough, **we should try**
    -   `predict_slot_dist`: by default we predict `mu` and `sigma`, maybe we can just predict one `mu` per slot

About auto checkpointing in mid-epoch

-   I need to go to [here](https://github.com/PyTorchLightning/pytorch-lightning/blob/45f6a3b1758f88af7fd776915539800cbc0137a9/pytorch_lightning/trainer/connectors/checkpoint_connector.py#L137), then I'll be able to restore training loops.
-   To save also the training state, I need to go to [here](https://github.com/PyTorchLightning/pytorch-lightning/blob/45f6a3b1758f88af7fd776915539800cbc0137a9/pytorch_lightning/trainer/connectors/checkpoint_connector.py#L471) **This can be done by setting val_check_interval!!!**
