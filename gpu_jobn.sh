#!/bin/bash

#SBATCH --job-name=docs_ex3
#SBATCH --output=docs_ex3.out
#SBATCH --gpus=1                # this allocates 72 CPU cores
#SBATCH --ntasks-per-gpu=3
#SBATCH --time=00:5:00

module load cray-python

srun python3 python_test.py