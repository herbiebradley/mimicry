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


class CustomCGANPDGenerator32(cgan_pd.CGANPDGenerator32):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        self.zero_grad()

        # Get batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and labels
        fake_images, fake_class_labels = self.generate_images_with_labels(
            num_images=batch_size, device=device)

        # Compute output logit of D thinking image real
        output = netD(fake_images, fake_class_labels)

        # Compute loss and backprop
        errG = self.compute_gan_loss(output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        log_data.add_metric('errG', errG, group='loss')

        return log_data


class CustomCGANPDDiscriminator32(cgan_pd.CGANPDDiscriminator32):
    def __init__(self, grad_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.resnet = resnet18(pretrained=True).to(device)
        self.l2_loss = nn.MSELoss().to(device)
        self.grad_lambda = grad_lambda

    def init_resnet(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        self.zero_grad()

        real_images, real_class_labels = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce logits for real images
        output_real = self.forward(real_images, real_class_labels)

        # Produce fake images and labels
        fake_images, fake_class_labels = netG.generate_images_with_labels(
            num_images=batch_size, device=device)
        fake_images, fake_class_labels = fake_images.detach(), fake_class_labels.detach()

        # Produce logits for fake images
        output_fake = self.forward(fake_images, fake_class_labels)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        real_images_detached = real_images.detach().to(device).requires_grad_(True)
        fake_images_detached = fake_images.detach().to(device).requires_grad_(True)

        # self.init_resnet(self.resnet)
        fake_resnet_logits = self.resnet(fake_images_detached)
        real_resnet_logits = self.resnet(real_images_detached)
        real_grad = autograd.grad(outputs=real_resnet_logits,
                                  inputs=real_images_detached,
                                  grad_outputs=torch.ones_like(real_resnet_logits, device=device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        fake_grad = autograd.grad(outputs=fake_resnet_logits,
                                  inputs=fake_images_detached,
                                  grad_outputs=torch.ones_like(fake_resnet_logits, device=device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        # fake_grad = fake_grad.view(fake_grad.size(0), -1)
        # grad_loss = -fake_grad.norm(2, dim=1).mean()
        grad_loss = self.l2_loss(real_grad, fake_grad)
        errD_total = errD + self.grad_lambda * grad_loss
        # Backprop and update gradients
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')
        return log_data


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
    netG = CustomCGANPDGenerator32(num_classes=10).to(device)
    netD = CustomCGANPDDiscriminator32(num_classes=10, grad_lambda=args.grad_lambda).to(device)
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

    generate_images(args, netG, device)
