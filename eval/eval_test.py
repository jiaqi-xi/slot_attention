import os
import argparse
import torch
import metrics
import numpy as np


@torch.no_grad()
def main(args):
    # frontend, model = train.build_model(args)
    # print(f'frontend = {frontend}')
    mask_path = os.path.join(args.dataset_root_dir, 'masks', args.split_name)
    mask1 = torch.as_tensor(np.load(os.path.join(mask_path, f'CLEVR_{args.split_name}_000001.npy'))).unsqueeze(0)
    mask2 = torch.as_tensor(np.load(os.path.join(mask_path, f'CLEVR_{args.split_name}_000002.npy'))).unsqueeze(0)
    
    print('mask1.shape = ', mask1.shape)

    mask1 = mask1.flatten(start_dim=2).transpose(-1, -2) # [batch_size, H*W, #group]
    mask2 = mask2.flatten(start_dim=2).transpose(-1, -2)
    print('mask1.shape changes to: ', mask1.size())

    mask_1 = torch.cat((mask1, mask2), 0)
    mask_2 = torch.cat((mask1, mask1), 0)
    print('mask_1.shape = ', mask_1.shape)

    ari = []
    ari.extend(metrics.adjusted_rand_index(mask_1, mask_2).tolist())
    print('Adjusted Rand Index:', ari, float(torch.tensor(ari).mean()))

    miou= []
    miou.append(metrics.mean_iou(mask_1, mask_2))
    print('mIOU: ', miou)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root-dir', default='/scratch/ssd004/scratch/jiaqixi/data/output')
    parser.add_argument('--split-name', default='train')
    args = parser.parse_args()

    main(args)
