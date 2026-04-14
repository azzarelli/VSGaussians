#!/bin/bash

#SBATCH --job-name=MipSplat3
#SBATCH --output=MipSplat_3%a.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=03:00:00
#SBATCH --array=1-3

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv

SAVEDIR=$HOME/data/studio_test5/scene3

ARGS="baseline.py"
EXP_NAME="MipSplat{$SLURM_ARRAY_TASK_ID}"

echo "Running with:"
echo "  CONFIG: $ARGS"
echo "  EXP_NAME: $EXP_NAME"
echo "  SUBSET: $SLURM_ARRAY_TASK_ID"

python gui.py -s "$SAVEDIR" \
  --expname "$SAVEDIR/$EXP_NAME" \
  --configs "arguments/$ARGS" \
  --test_iterations 2000 \
  --subset $SLURM_ARRAY_TASK_ID
