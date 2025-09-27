ModelHiddenParams = dict(
    scene_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [512, 100],
     'wavelevel':2
    },
    target_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [512, 100],
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
    batch_size=1, # Was 4
    
    opacity_reset_interval = 120000,    
    densification_interval = 500,
    
    densify_from_iter = 3000,
    densify_until_iter = 12000,
    
    
)