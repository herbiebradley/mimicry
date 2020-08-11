#!/bin/bash
# Arg1: Name of run/model, will be used as log dir name.

python train_GAN.py --run "${1}"
python fid/fid_score.py --log_dir "log/${1}/generated_images" --gpu "0" --output_name "${1}"
python cifar10_train.py --run "${1}" --data_dir "log/${1}/generated_images"
