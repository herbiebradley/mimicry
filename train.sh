#!/bin/bash
# Arg1: Log dir
# Arg2: Name of approach to use when logging results

python train_GAN.py --log_dir ${1}
python fid/fid_score.py --log_dir "${1}/generated_images" --gpu "0" --output_name "${2}"
python cifar10_train.py --run "${2}"
