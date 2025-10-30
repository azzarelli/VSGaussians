#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --output=test_run.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=08:00:00

ARGS=$1
EXP_NAME=$2

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv

SAVEDIR=$HOME/data/studio4-1/
echo "Training starting..."
python gui.py -s "$SAVEDIR" \
  --expname "$SAVEDIR/$EXP_NAME" \
  --configs arguments/$ARGS \
  --test_iterations 4000
