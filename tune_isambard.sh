#!/bin/bash

declare -a CONFIGS=("studio4.py")
# "lam01.py" "lam0075.py" "lam005.py" "lam001.py")
declare -a EXPNAMES=("baseline")
# "lam01" "lam0075" "lam005" "lam001")

for i in "${!CONFIGS[@]}"; do
  echo "Submitting job for ${CONFIGS[$i]}..."
  sbatch gpu_jobn.sh "${CONFIGS[$i]}" "${EXPNAMES[$i]}"
done