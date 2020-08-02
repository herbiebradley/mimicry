import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision

from fid.fid_score import calculate_fid_given_paths
from fid.inception import InceptionV3
from torch_mimicry.nets import sngan


def generate_fid_images(args):
    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(args.log_dir, 'checkpoints', 'netG')

    if not os.path.exists(ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                ckpt_dir))
    ckpt_file = os.path.join(ckpt_dir, f'netG_{args.iter}_steps.pth')

    netG = sngan.SNGANGenerator32().to(device)
    netG.restore_checkpoint(ckpt_file=ckpt_file, optimizer=None)

    with torch.no_grad():
        # Set model to evaluation mode
        netG.eval()

        fid_img_num = 50000
        print_every = 20
        batch_size = min(fid_img_num, args.batch_size)
        generated_images_dir = os.path.join(args.log_dir, "generated_images")
        if not os.path.exists(generated_images_dir):
            os.makedirs(generated_images_dir)

            # Collect all samples()
            start_time = time.time()
            img_count = 0
            for idx in range(fid_img_num // batch_size):
                # Collect fake image
                fake_images = netG.generate_images(num_images=batch_size,
                                                device=device).detach().cpu()

                for img_idx in range(batch_size):
                    torchvision.utils.save_image(fake_images[img_idx],
                                                f'{generated_images_dir}/fake_samples_{img_count:05d}.png',
                                                normalize=True, padding=0)
                    img_count += 1

                # Print some statistics
                if (idx + 1) % print_every == 0:
                    end_time = time.time()
                    print(
                        "INFO: Generated image {}/{} ({:.4f} sec/idx)"
                        .format(
                            (idx + 1) * batch_size, fid_img_num,
                            (end_time - start_time) / (print_every * batch_size)))
                    start_time = end_time

    return generated_images_dir


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

    generated_images_dir = generate_fid_images(args)

    paths = [generated_images_dir, "fid/fid_stats_cifar10_train.npz"]
    fid_value = calculate_fid_given_paths(paths,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims)

    log_output = open(f"{args.log_dir}/log_FID.txt", 'a')
    print(f'{args.output_name} / Iter {args.iter:06d} : {fid_value:.4f}', file=log_output)
    log_output.close()
    print('FID: ', fid_value)
