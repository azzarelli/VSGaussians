#!/bin/bash

#SBATCH --job-name=set2.1
#SBATCH --output=set2.1.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=02:30:00

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv

SAVEDIR=$HOME/data/studio_test5/scene2

EXP_NAME="set1"

echo "Running with:"
echo "  EXP_NAME: $EXP_NAME"

python gui.py -s "$SAVEDIR" \
  --expname "$SAVEDIR/$EXP_NAME" \
  --configs "arguments/baseline.py" \
  --test_iterations 2000 \
  --test-frames 23 \
  --subset 1
