#!/usr/bin/env bash

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch











python download_data.py --data train --noise real
python generate_patches_sidd.py

cd ..
./train.sh Denoising/Options/RealDenoising_Restormer.yml