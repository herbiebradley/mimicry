import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils import data as tdata

import torch_mimicry as mmc
from torch_mimicry.nets import cgan_pd


def generate_images(args, netG):
    ckpt_file = os.path.join(args.log_dir, 'checkpoints', 'netG', f'netG_{args.iter}_steps.pth')
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', type=str, default='./log/cgan1',
                        help='Log directory for run.')
    parser.add_argument('-c', '--gpu', default='0', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--iter", default=100000, type=int,
                        help='Training step iteration, to use when logging results')

    args = parser.parse_args()

    # Data handling objects
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='./data', name='cifar10')
    dataloader = tdata.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    # Define models and optimizers
    netG = cgan_pd.CGANPDGenerator32(num_classes=10).to(device)
    netD = cgan_pd.CGANPDDiscriminator32(num_classes=10).to(device)
    optD = optim.Adam(netD.parameters(), 0.0002, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 0.0002, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=args.iter,
        lr_decay='linear',
        dataloader=dataloader,
        log_dir=args.log_dir,
        device=device)
    trainer.train()

    generate_images(args, netG)
