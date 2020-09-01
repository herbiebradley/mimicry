#!/bin/bash
# Arg1: Name of run/model, will be used as log dir name.
# Arg2: Grad lambda.

# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
set -euo pipefail

python train_latent.py --run "${1}" --iter 100000
# python calculate_fid.py --log_dir "log/${1}-${2}" --output_name "${1}-${2}"
# python cifar10_train.py --run "${1}" --data_dir "log/${1}/generated_images"
