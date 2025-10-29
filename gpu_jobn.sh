#!/bin/bash

#SBATCH --job-name=docs_ex3
#SBATCH --output=docs_ex3.out
#SBATCH --gpus=1                # this allocates 72 CPU cores
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00:5:00

module load cuda/11.8

source ~/miniforge3/bin/activate
conda activate vres
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# SAVEDIR=/studio4-1/studio4-1/
# EXP_NAME=test

# echo "Training starting..."
# srun python gui.py -s "$SAVEDIR" \
#   --expname "$SAVEDIR/$EXP_NAME" \
#   --configs arguments/studio4.py \
#   --test_iterations 2000
