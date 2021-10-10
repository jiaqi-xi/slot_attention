# Slot Attention on CLEVR Video dataset

## Prerequisites

-   PyTorch 1.9.1 with CUDA 11.1
-   PyTorch-Lightning 1.4.9
-   wandb for logging and visualization
-   other packages, see the error output if needed

After `pip install wandb`, remember to log in so that you can monitor your experiments results on the webpage. First go to https://wandb.ai to sign up an account, then use `wandb login` to login.

## Data Preparation

Unzip `clevr.tar.gz` to your `DATA_DIR`, then modify [this line](https://github.com/Wuziyi616/slot_attention/blob/8dea9335d34fe95f6ef5d7903a065b99970de549/clevr_video/params.py#L15) to that path.

-   `images/` are videos in `avi` format
-   `scenes/` are their corresponding annotations, e.g. how many objects in each video

## Per frame slot attention

To train the model, simply run:

```
python train.py --params params.py
```

`params.py` is the config file you are using. If you want to accelerate experiments with mixed precision training, simply add the `--fp16` flag to the command.

Note: currently multi-GPU training is not supported (actually you can do that by modifying `gpus` in the `params.py` file, but the trained results seem in bad quality, don't know why)

The weights will be saved in `checkpoint/${params}`, only the best 3 are kept. You can access the training logs at https://wandb.ai/home.

To test a checkpoint, simply run:

```
python test.py --params params.py --weight 'path/to/weight'
```

By default, two videos in `mp4` format (one from training set, one from val set) will be saved in `path/to/weight/vis/`
