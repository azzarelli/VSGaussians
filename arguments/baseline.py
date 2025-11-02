ModelHiddenParams = dict(
)

OptimizationParams = dict(
    batch_size=2, # Was 4
    dataloader=True,
    iterations=16000,
    
    prune_opa=0.005,
    grow_grad2d=0.0001,
    grow_scale3d=0.01,
    grow_scale2d=0.1,
    prune_scale3d=0.2,

    densify_from_iter = 100,
    densify_until_iter = 3000,
    opacity_reset_interval = 3000,    
    densification_interval = 100,
    
)