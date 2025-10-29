#!/bin/bash

#SBATCH --job-name=docs_ex1
#SBATCH --output=docs_ex1.out
#SBATCH --gpus=1
#SBATCH --time=00:5:00         # Hours:Mins:Secs

module load cuda/11.8
module load cudatoolkit/24.11_11.8

hostname
nvidia-smi --list-gpus
