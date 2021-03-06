import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch import autograd
from torch.utils import data as tdata

import torch_mimicry as mmc
from cifar10_models import resnet18
from torch_mimicry.nets import cgan_pd
from utils.gan_utils import parse_args


def init_resnet(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def get_indices(dataset, class_name):
    """Return indices of class_name in dataset."""
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(name=args.run, project='gans', config=args)

    # Data handling objects
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='data/', name='cifar10')
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
        num_steps=args.iters,
        lr_decay=args.lr_decay,
        dataloader=dataloader,
        log_dir=args.log_dir,
        device=device,
        log_steps=args.log_steps,
        wandb=False)
    trainer.train()

    # Get checkpoint file and restore it.
    ckpt_file = os.path.join(args.log_dir, 'checkpoints', 'netG', f'netG_{args.iters}_steps.pth')
    generated_images_dir = os.path.join(args.log_dir, "generated_images")
    os.makedirs(generated_images_dir, exist_ok=True)
    netG.restore_checkpoint(ckpt_file=ckpt_file)
    for class_label in range(10):
        os.makedirs(os.path.join(generated_images_dir, f'{class_label}'), exist_ok=True)

    # Init gradient difference loss and residual network.
    l2_loss = nn.MSELoss().to(device)
    resnet = resnet18().to(device)
    for param in resnet.parameters():
        param.requires_grad = False

    for class_label in range(10):
        print(f'Beginning Latent Descent for class index: {class_label}...')
        # Get the indices of the images with the label 'class_label' in CIFAR10.
        idx = get_indices(dataset, class_label)
        # Load the indices into a torch.utils.data.DataLoader
        dataloader = tdata.DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=8, pin_memory=True, drop_last=True,
                                  sampler=tdata.sampler.SubsetRandomSampler(idx))
        iter_loader = iter(dataloader)
        # Sample random noise and define optimiser.
        noise = torch.randn((args.batch_size, 128), device=device, requires_grad=True)
        label = torch.tensor(class_label, device=device).reshape(1)
        opt_latent = optim.Adam([noise], lr=0.001)

        for iteration in range(args.latent_iters):
            opt_latent.zero_grad()
            # Get batch of data.
            try:
                real_batch = next(iter_loader)
            except StopIteration:
                print
                iter_loader = iter(dataloader)
                real_batch = next(iter_loader)
            real_batch = real_batch[0].to(device, non_blocking=True).requires_grad_(True)
            # Generate image from noise.
            fake_batch = netG.forward(noise, label)

            # Pass real and fake images through resnet.
            init_resnet(resnet)
            fake_resnet_logits = resnet(fake_batch)
            real_resnet_logits = resnet(real_batch)
            # Get gradients of resnet outputs with respect to the real and fake images.
            real_grad = autograd.grad(outputs=real_resnet_logits,
                                  inputs=real_batch,
                                  grad_outputs=torch.ones_like(real_resnet_logits, device=device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
            fake_grad = autograd.grad(outputs=fake_resnet_logits,
                                    inputs=fake_batch,
                                    grad_outputs=torch.ones_like(fake_resnet_logits, device=device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
            # Gradient difference loss
            grad_loss = l2_loss(real_grad, fake_grad)
            # Gradient magnitude loss.
            fake_grad = fake_grad.view(fake_grad.size(0), -1)
            fake_grad_loss = -fake_grad.norm(2, dim=1).mean()
            # L_2 loss + lambda * magnitude loss.
            grad_loss = grad_loss + args.grad_lambda * fake_grad_loss
            grad_loss.backward()
            opt_latent.step()
            wandb.log({'grad_loss': grad_loss,
                       'grad_magnitude': fake_grad_loss},
                      step=(class_label * args.latent_iters) + iteration)

            if iteration % 1000 == 0:
                print(f'At iteration {iteration} out of {args.latent_iters} for class index {class_label}.')

            if iteration >= (args.latent_iters - 100):
                for img_count in range(50):
                    img_path = os.path.join(generated_images_dir, str(class_label),
                                            f'{((iteration-(args.latent_iters - 100)) * 50 + img_count):04d}.png')
                    torchvision.utils.save_image(fake_batch[img_count], img_path, normalize=True, padding=0)
