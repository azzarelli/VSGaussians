#!/bin/bash

#SBATCH --job-name=docs_ex3
#SBATCH --output=docs_ex3.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00:10:00

module load cuda/11.8

source ~/miniforge3/bin/activate
# conda create -n vsenv python=3.10 -y
conda activate vsenv

# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.0+nv23.10 torchvision==0.16.0+nv23.10 \
    --extra-index-url https://pypi.nvidia.com
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# SAVEDIR=/studio4-1/studio4-1/
# EXP_NAME=test

# echo "Training starting..."
# srun python gui.py -s "$SAVEDIR" \
#   --expname "$SAVEDIR/$EXP_NAME" \
#   --configs arguments/studio4.py \
#   --test_iterations 2000
