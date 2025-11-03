#!/bin/bash

SAVEDIR=$1
EXP_NAME=$2

# Try to determine base path automatically
if [ -d "/media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR" ]; then
  BASEDIR="/media/barry/56EA40DEEA40BBCD/DATA"
elif [ -d "/data/$SAVEDIR" ]; then
  BASEDIR="/data"
else
  echo "Error: Could not find data directory for $SAVEDIR"
  exit 1
fi

if [ "$3" == "view" ]; then
  echo "Viewing..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s "$BASEDIR/$SAVEDIR/" \
    --expname "$SAVEDIR/$EXP_NAME" \
    --configs arguments/baseline.py \
    --start_checkpoint "$4" --view-test
else
  echo "Training starting..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s "$BASEDIR/$SAVEDIR/" \
    --expname "$SAVEDIR/$EXP_NAME" \
    --configs arguments/baseline.py \
    --test_iterations 2000
fi
