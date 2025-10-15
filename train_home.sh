#!/bin/bash

# Assign the first argument to a variable
SAVEDIR=$1
EXP_NAME=$2

ARGS=default.py

if [ "$3" == "render" ]; then
  echo "Rendering process starting..."
  python render.py --model_path "output/home/$SAVEDIR/$EXP_NAME" --skip_train --configs arguments/dynerf/steak.py

elif [ "$3" == "view" ]; then
  echo "Viewing..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "home/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/steak.py --test_iterations 1000 --start_checkpoint $4 --view-test --bundle-adjust
elif [ "$3" == "ext" ]; then
  echo "Extending..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "home/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/steak.py --test_iterations 1000 --start_checkpoint 7000 
elif [ "$3" == "skip-coarse" ]; then
  echo "Skip Coarse..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "home/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/steak.py --test_iterations 1000 --skip-coarse $4

elif [ "$3" == "nvs" ]; then
  echo "NVS..."

  CUDA_LAUNCH_BLOCKING=1 python nvs.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "home/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/steak.py --test_iterations 1000 --start_checkpoint $4 --view-test

else
  echo "Training starting..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "home/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/steak.py --test_iterations 1000 --bundle-adjust --downsample 2
fi
