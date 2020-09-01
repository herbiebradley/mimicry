import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import wandb
from torch import autograd
from torch.utils import data as tdata

import torch_mimicry as mmc
from cifar10_models import resnet18
from torch_mimicry.nets import cgan_pd
from utils.gan_utils import generate_images, parse_args


def init_resnet(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


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
        num_steps=args.iter,
        lr_decay=args.lr_decay,
        dataloader=dataloader,
        log_dir=args.log_dir,
        device=device,
        log_steps=args.log_steps)
    trainer.train()

    ckpt_file = os.path.join(args.log_dir, 'checkpoints', 'netG', f'netG_{args.iter}_steps.pth')
    netG.restore_checkpoint(ckpt_file=ckpt_file)
    resnet = resnet18().to(device)
    latent_iters = 5

    for param in resnet.parameters():
        param.requires_grad = False

    noise = torch.randn((args.batch_size, 128), device=device, requires_grad=True)
    label = torch.tensor(0, device=device).reshape(1)
    opt_latent = optim.Adam([noise], lr=0.001)

    for iter in range(latent_iters):
        opt_latent.zero_grad()
        init_resnet(resnet)
        fake_batch = netG.forward(noise, label)
        resnet_output = resnet(fake_batch)

        fake_grad = autograd.grad(outputs=resnet_output,
                                  inputs=fake_batch,
                                  grad_outputs=torch.ones_like(resnet_output, device=device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        fake_grad = fake_grad.view(fake_grad.size(0), -1)
        grad_loss = -fake_grad.norm(2, dim=1).mean()
        grad_loss.backward()
        opt_latent.step()
        wandb.log({'grad_loss': grad_loss}, step=iter)
