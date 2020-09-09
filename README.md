# MSc Project Code

The code in the `cifar10_models` folder, `cifar10_download.py`, `cifar10_module.py`, and `cifar10_train.py` is modified from https://github.com/huyvnphan/PyTorch_CIFAR10.

Thanks to https://github.com/mseitzer/pytorch-fid/ for the code in the `fid` folder.

The code in the `torch_mimicry` folder is a modified version of https://github.com/kwotsin/mimicry. LSGAN loss, wandb logging, and data loading from pinned memory have been added.

Everything else in this repository is my own work.

## System requirements

- Python 3
- CPU or NVIDIA GPU + CUDA

## Installation

It is recommended to run `bash setup.sh` to install requirements, especially if in a Docker container. This script also installs Pillow-SIMD for faster image processing, and downloads the CIFAR10 statistics for calculating the FID. Alternatively, `pip install -r requirements.txt` will install all the necessary Python packages. `wandb login` should be run before doing anything else, since this will ensure all runs are logged in the cloud at `wandb.com`.

The two lines at the end of the script can be uncommented if a pre-trained residual network needs to be used.

## Using this repo

`bash train.sh` will train a GAN for 50,000 iterations, generate data, calculate the FID, and train a residual network on the generated data. The script has two mandatory arguments: first the name of the run and second the value for the lambda parameter in the gradient difference loss. The gradient magnitude loss can be enabled by uncommenting 2 lines in `train_GAN.py`. A pre-trained residual network in the loss can be enabled by changing `pretrained=True` on line 60 of `train_GAN.py`, after downloading the weights as described above. An example of running the script: `bash train.sh Grad-Diff 1.5` will save the GAN data and logs in `log/Grad-Diff` and use lambda = 1.5 in the loss.

`bash latent.sh` will train a GAN for 100,000 iterations, save it in `log/Latent/`, perform gradient descent in the latent space, generate data, calculate the FID, and train a residual network on the generated data. The script has three mandatory arguments: first the name of the run, second the number of iterations to perform gradient descent in the latent space, and third the value for lambda in the loss. Lambda here is equivalent to beta in the dissertation, with alpha = 1. Example of running the script: `bash latent.sh Grad-Mag-0.1 10000 0.1`. This will save the data to `log/Latent/` but create a `wandb` run with the name `Grad-Mag-0.1`.

Most hyperparameters can be changed by the command line, see `utils/gan_utils.py` for full details.
