#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --output=baseline.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=03:30:00

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv

SAVEDIR=$HOME/data/studio4-1/

ARGS="baseline.py"
EXP_NAME="baseline"


echo "Running with:"
echo "  CONFIG: $ARGS"
echo "  EXP_NAME: $EXP_NAME"

python gui.py -s "$SAVEDIR" \
  --expname "$SAVEDIR/$EXP_NAME" \
  --configs "arguments/$ARGS" \
  --test_iterations 2000 \
  --num-cams $3 \
  --num-textures $4 \
  --num-textures-block $5