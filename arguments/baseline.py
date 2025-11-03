ModelHiddenParams = dict(
)

OptimizationParams = dict(
    batch_size=2, # Was 4
    dataloader=True,
    iterations=16000,
    
    densify_from_iter = 100,
    densification_interval = 100,
)