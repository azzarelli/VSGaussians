#!/bin/bash

declare -a CONFIGS=("studio4.py")
# "lam01.py" "lam0075.py" "lam005.py" "lam001.py")
declare -a EXPNAMES=("baseline")
# "lam01" "lam0075" "lam005" "lam001")

for i in "${!CONFIGS[@]}"; do
  CONFIG=${CONFIGS[$i]}
  EXP=${EXPNAMES[$i]}

  echo "Submitting job for $CONFIG..."
  sbatch --export=ALL,ARGS="$CONFIG",EXP_NAME="$EXP" gpu_jobn.sh
done