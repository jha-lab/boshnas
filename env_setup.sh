#!/bin/sh

# Script to install required packages in conda
# Author : Shikhar Tuli

module load anaconda3

# Create a new conda environment
conda create --name boshnas python=3.6 pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.2 -c pytorch -c nvidia

conda activate boshnas

# Install dependencies
cd naszilla
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install -e .

conda install -c omgarcia gcc-6
conda install libgcc

export LD_LIBRARY_PATH=/home/$USER/.conda/envs/boshnas/lib/:$LD_LIBRARY_PATH

# Run these if you get GCC errors
# mv ~/.conda/envs/boshnas/lib/libstdc++.so.6 ~/.conda/envs/boshnas/lib/libstdc++.so.6.bak
# mv ~/.conda/envs/boshnas/lib/libstdc++.so.6.0.26 ~/.conda/envs/boshnas/lib/libstdc++.so.6

pip install torch-sparse
