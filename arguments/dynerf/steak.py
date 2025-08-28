ModelHiddenParams = dict(
    scene_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [512, 25],
     'wavelevel':2
    },
    target_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [512, 50],
     'wavelevel':2
    },

    net_width = 128,
    
    plane_tv_weight = 0.001,
    time_smoothness_weight = 0.0001,
    l1_time_planes =  0.001,
    minview_weight=0.001,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=16000, # 7000 salmon with 4 batch, 8000 with flame steak
    batch_size=2, # Was 4
    
    opacity_reset_interval = 3000,    

    pruning_interval = 100,
    pruning_from_iter=3000,
    lambda_dssim = 0., #0.1,
    
)