import argparse
import os

import numpy as np
import torch
import torchvision.utils as vutils


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', type=str, default='',
                        help='Log directory for run.')
    parser.add_argument('-c', '--gpu', default='0', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--iter", default=50000, type=int,
                        help='Number of training iterations.')
    parser.add_argument("--latent_iters", default=10000, type=int,
                        help='Number of iterations to perform gradient descent'
                        'in the latent space.')
    parser.add_argument('--lr_D', type=float, default=0.0002,
                        help='Discriminator learning rate')
    parser.add_argument('--lr_G', type=float, default=0.0002,
                        help='Generator learning rate')
    parser.add_argument('--lr_decay', type=str, default='linear',
                        help='The learning rate decay policy to use.')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Adam beta 1')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='Adam beta 2')
    parser.add_argument('--grad_lambda', type=float, default=1.0,
                        help='Lambda multiplier for grad loss term.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of iterations of D for each G iteration.')
    parser.add_argument('--log_steps', type=int, default=20,
                        help='Number of training steps before writing summaries'
                        'to WandB.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', type=str, default='', help="Run Name")
    args = parser.parse_args()
    if args.run == '':
        args.run = args.log_dir[4:-1]
    elif args.log_dir == '':
        args.log_dir = f'log/{args.run}'
    return args


def generate_images(args, netG, device):
    ckpt_file = os.path.join(args.log_dir, 'checkpoints', 'netG', f'netG_{args.iter}_steps.pth')
    generated_images_dir = os.path.join(args.log_dir, "generated_images")
    if not os.path.isfile(ckpt_file) or os.path.exists(generated_images_dir):
        print("INFO: No data generated.")
        return
    os.makedirs(generated_images_dir, exist_ok=True)
    netG.restore_checkpoint(ckpt_file=ckpt_file)
    for class_label in range(10):
        os.makedirs(os.path.join(generated_images_dir, f'{class_label}'), exist_ok=True)

    with torch.no_grad():
        netG.eval()

        dset_size = 50000
        class_labels = {key: 5000 for key in range(10)}

        for idx in range(dset_size):
            noise = torch.randn((1, 128), device=device)
            label = torch.tensor(np.random.choice(list(class_labels.keys())), device=device).reshape(1)
            fake_image = netG.forward(noise, label)

            label_int = int(label.item())
            class_labels[label_int] -= 1
            img_path = os.path.join(generated_images_dir, str(label_int), f'{(5000-class_labels[label_int]):04d}.png')
            vutils.save_image(fake_image, img_path, normalize=True, padding=0)
            class_labels = {key: class_labels[key] for key in class_labels.keys() if class_labels[key] != 0}

        print("INFO: Generated data saved.")
