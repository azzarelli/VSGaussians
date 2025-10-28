#!/bin/bash

# Assign the first argument to a variable
SAVEDIR=$1
EXP_NAME=$2

ARGS=default.py

if [ "$3" == "view" ]; then
  echo "Viewing..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4.py --start_checkpoint $4 --view-test
else
  echo "Training starting..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4.py --test_iterations 2
fi
