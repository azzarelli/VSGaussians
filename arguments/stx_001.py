ModelHiddenParams = dict(
    target_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [256, 200],
     'wavelevel':2
    },

    net_width = 128,
    
    plane_tv_weight = 0.001,
    time_smoothness_weight = 0.0001,
    l1_time_planes =  0.001,
    minview_weight=0.001,
)

OptimizationParams = dict(
    batch_size=2, # Was 4

    dataloader=True,
    iterations=16000, # 7000 salmon with 4 batch, 8000 with flame steak
    
    lambda_dssim = 0.2,
    lambda_dist = 0.0,
    lambda_normal = 0.001,
    lambda_alpha=1.,
    lambda_inv=0.1,
    
    opacity_lr =0.01,

    lambda_lr = 0.0025,# 0.0025,
    tex_mu_lr = 0.0025,
    tex_s_lr = 0.001,

    densify_from_iter = 100,
    densify_until_iter = 3000,
    opacity_reset_interval = 3000,    
    densification_interval = 100,
    
)