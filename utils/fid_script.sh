#!/bin/bash
# Arg1: Log dir
# Arg2: Name of approach to use when logging results

python ../calculate_fid.py --log_dir "${1}" --gpu "0" --output_name "${2}"
