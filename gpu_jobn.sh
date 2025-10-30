#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --output=test_run.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00:01:00

module load cray-python
module load cudatoolkit

source ~/miniforge3/bin/activate
conda activate vsenv

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

SAVEDIR=$HOME/data/studio4-1/
echo "Running with:"
echo "  CONFIG: $ARGS"
echo "  EXP_NAME: $EXP_NAME"

echo "Training starting..."
srun python gui.py -s "$SAVEDIR" \
  --expname "$SAVEDIR/$EXP_NAME" \
  --configs "arguments/$ARGS" \
  --test_iterations 4000