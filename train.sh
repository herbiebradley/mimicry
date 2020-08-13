#!/bin/bash
# Arg1: Name of run/model, will be used as log dir name.
# Arg2: Grad lambda.

python train_GAN.py --run "${1}-${2}" --grad_lambda "${2}"
python calculate_fid.py --log_dir "log/${1}-${2}" --output_name "${1}"
python cifar10_train.py --run "${1}-${2}" --data_dir "log/${1}/generated_images"
