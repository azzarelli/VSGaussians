#!/bin/bash

#SBATCH --job-name=C2_3.3
#SBATCH --output=C2_3.3.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=03:30:00

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv

SAVEDIR=$HOME/data/studio_test5/scene3/

ARGS="baseline.py"
EXP_NAME="C2_3.3"

echo "Running with:"
echo "  CONFIG: $ARGS"
echo "  EXP_NAME: $EXP_NAME"

python gui.py -s "$SAVEDIR" \
  --expname "$SAVEDIR/$EXP_NAME" \
  --configs "arguments/$ARGS" \
  --test_iterations 2000 \
  --subset 3