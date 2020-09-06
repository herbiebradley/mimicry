#!/bin/bash
# Arg1: Name of run/model.
# Arg2: Number of iterations to perform gradient descent in the latent space.
# Arg3: Grad Lambda/

# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
set -euo pipefail

rm -rf log/Latent1/generated_images
python train_latent.py --run "${1}" --log_dir "log/Latent1" --iters 100000 --latent_iters ${2} --grad_lambda ${3}
python calculate_fid.py --log_dir "log/Latent1" --output_name "${1}-${2}"
python cifar10_train.py --run "${1}" --data_dir "log/Latent1/generated_images"
