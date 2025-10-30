#!/bin/bash

#SBATCH --job-name=docs_ex3
#SBATCH --output=docs_ex3.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00:40:00


module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip install wheel
TORCH_CUDA_ARCH_LIST="9.0" CC=gcc-13 CXX=g++-13 pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat


python python_test.py


# SAVEDIR=/studio4-1/studio4-1/
# EXP_NAME=test
# echo "Training starting..."
# srun python gui.py -s "$SAVEDIR" \
#   --expname "$SAVEDIR/$EXP_NAME" \
#   --configs arguments/studio4.py \
#   --test_iterations 2000
