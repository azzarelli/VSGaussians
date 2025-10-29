#!/bin/bash

#SBATCH --job-name=docs_ex1
#SBATCH --output=docs_ex1.out
#SBATCH --gpus=1
#SBATCH --time=00:5:00         # Hours:Mins:Secs

hostname
nvidia-smi --list-gpus
