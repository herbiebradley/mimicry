#!/bin/bash
# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
set -euo pipefail

pip install -r requirements.txt

pip uninstall -y Pillow

apt install -y \
    libjpeg-turbo8-dev \
    zlib1g-dev \
    libtiff5-dev \
    liblcms2-dev \
    libfreetype6-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libopenjp2-7-dev \
    libraqm0 \

CC="cc -mavx2" pip install --no-cache-dir -U -I --force-reinstall pillow-simd \
    --global-option="build_ext" \
    --global-option="--enable-zlib" \
    --global-option="--enable-jpeg" \
    --global-option="--enable-tiff" \
    --global-option="--enable-freetype" \
    --global-option="--enable-lcms" \
    --global-option="--enable-webp" \
    --global-option="--enable-webpmux" \
    --global-option="--enable-jpeg2000" \

python -c "from PIL import __version__; print(f'Pillow Version: {__version__}')"

wget http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz -P fid/
