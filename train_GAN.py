import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
import wandb
from torch.utils import data as tdata

import torch_mimicry as mmc
from torch_mimicry.nets import cgan_pd

# TODO: Add new loss function


class CustomCGANPDGenerator32(cgan_pd.CGANPDGenerator32):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        pass


class CustomCGANPDDiscriminator32(cgan_pd.CGANPDDiscriminator32):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        pass


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Log directory for run.')
    parser.add_argument('-c', '--gpu', default='0', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--iter", default=100000, type=int,
                        help='Number of training iterations.')
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
        args.run = args.log_dir
    return args


def generate_images(args, netG):
    ckpt_file = os.path.join(args.log_dir, 'checkpoints', 'netG', f'netG_{args.iter}_steps.pth')
    if not os.path.isfile(ckpt_file):
        print("INFO: No data generated.")
        return
    netG.restore_checkpoint(ckpt_file=ckpt_file)
    generated_images_dir = os.path.join(args.log_dir, "generated_images")
    os.makedirs(generated_images_dir, exist_ok=True)
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


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(name=args.run, project='gans', config=args)

    # Data handling objects
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='./data', name='cifar10')
    dataloader = tdata.DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)

    # Define models and optimizers
    netG = cgan_pd.CGANPDGenerator32(num_classes=10).to(device)
    netD = cgan_pd.CGANPDDiscriminator32(num_classes=10).to(device)
    optD = optim.Adam(netD.parameters(), args.lr_D, betas=(args.beta1, args.beta2))
    optG = optim.Adam(netG.parameters(), args.lr_G, betas=(args.beta1, args.beta2))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=args.iter,
        lr_decay=args.lr_decay,
        dataloader=dataloader,
        log_dir=args.log_dir,
        device=device,
        log_steps=args.log_steps)
    trainer.train()

    generate_images(args, netG)
