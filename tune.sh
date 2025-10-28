SAVEDIR=studio_test4-1/pose_data


# EXP_NAME=nomip_o01
# CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4_o01.py --test_iterations 2000

# EXP_NAME=nomip_o005
# CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4_o005.py --test_iterations 2000
# EXP_NAME=nomip_o001
# CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4_o001.py --test_iterations 2000

EXP_NAME=nomip_s001
CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4_s001.py --test_iterations 2000

EXP_NAME=nomip_s0005
CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/$SAVEDIR/ --expname "$SAVEDIR/$EXP_NAME" --configs arguments/studio4_s0005.py --test_iterations 2000
