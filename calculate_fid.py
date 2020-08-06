import argparse
import os
import random

import numpy as np
import torch

from fid.fid_score import calculate_fid_given_paths
from fid.inception import InceptionV3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', type=str, default='',
                        help='Log directory for run.')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='0', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--output_name", default="", type=str,
                        help='Name of method used, to use when logging results')
    parser.add_argument("--iter", default=100000, type=int,
                        help='Training step iteration, to use when logging results')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Make sure the random seeds are fixed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    generated_images_dir = os.path.join(args.log_dir, "generated_images")

    paths = [generated_images_dir, "fid/fid_stats_cifar10_train.npz"]
    fid_value = calculate_fid_given_paths(paths,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims)

    log_output = open(f"{args.log_dir}/log_FID.txt", 'a')
    print(f'{args.output_name} / Iter {args.iter:06d} : {fid_value:.4f}', file=log_output)
    log_output.close()
    print('FID: ', fid_value)
