#!/bin/bash
# Arg1: Folder to check for images
# Arg2: Name of approach to use when logging results

python ../fid/fid_score.py "${1}/generated_images" --gpu "0" --output_name "${2}"
