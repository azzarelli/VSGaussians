import os
import optix as ox
import cupy as cp
import numpy as np

# -------------------------------------------------------------------
# Logging callback
def log_callback(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))


# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Module and pipeline creation
def create_module(ctx, pipeline_opts, cuda_src):
    compile_opts = ox.ModuleCompileOptions(
        debug_level=ox.CompileDebugLevel.FULL,
        opt_level=ox.CompileOptimizationLevel.LEVEL_0
    )
    return ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp   = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp    = ox.ProgramGroup.create_hitgroup(ctx, module,
                                                 entry_function_CH="__closesthit__ch")
    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx,
                           compile_options=pipeline_options,
                           link_options=link_opts,
                           program_groups=program_grps)

    pipeline.compute_stack_sizes(1, 0, 1)  # (max_trace_depth, max_cc_depth, max_dc_depth)
    return pipeline


def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps
    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt   = ox.SbtRecord(miss_grp)       # no miss data needed
    hit_sbt    = ox.SbtRecord(hit_grp)        # no hit data needed
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt,
                                miss_records=miss_sbt,
                                hitgroup_records=hit_sbt)
    return sbt


# -------------------------------------------------------------------
# Ray/triangle intersection function
def ray_triangle_intersections(ctx, pipeline, sbt, gas,
                               ray_origins, ray_directions):
    """
    ray_origins:   (N,3) cupy.float32
    ray_directions:(N,3) cupy.float32
    triangles are already in the GAS
    Returns: (N,1) cupy.int32
    """
    N = ray_origins.shape[0]
    hit_ids = cp.full((N,), -1, dtype=cp.int32)

    # Build launch params
    params_tmp = [
        ('u8', 'ray_origins'),
        ('u8', 'ray_directions'),
        ('u8', 'hit_ids'),
        ('u4', 'num_rays'),
        ('u8', 'handle')
    ]
    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                   formats=[p[0] for p in params_tmp])
    params['ray_origins']    = ray_origins.data.ptr
    params['ray_directions'] = ray_directions.data.ptr
    params['hit_ids']        = hit_ids.data.ptr
    params['num_rays']       = N
    params['handle']         = gas.handle

    stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=(N,1,1), params=params, stream=stream)
    stream.synchronize()

    return hit_ids.reshape(N,1)


# -------------------------------------------------------------------
# Example test run
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    cuda_src = os.path.join(script_dir, "cuda", "triangle_adrian.cu")

    ctx = ox.DeviceContext(validation_mode=False,
                           log_callback_function=log_callback,
                           log_callback_level=3)

    # One triangle in z=0 plane
    triangles = cp.array([[[0.0,0.0,0.0],
                           [1.0,0.0,0.0],
                           [0.0,1.0,0.0]]], dtype=np.float32)

    
    M = triangles.shape[0]
    vertices = triangles.reshape(M*3, 3).astype(np.float32)
    vertices = cp.asarray(vertices)  # ensure on GPU
    build_input = ox.BuildInputTriangleArray(vertices, flags=[ox.GeometryFlags.NONE])
    gas = ox.AccelerationStructure(ctx, build_input, compact=True)

    pipeline_options = ox.PipelineCompileOptions(
        traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
        num_payload_values=1,
        num_attribute_values=2,
        exception_flags=ox.ExceptionFlags.NONE,
        pipeline_launch_params_variable_name="params"
    )

    module = create_module(ctx, pipeline_options, cuda_src)
    program_grps = create_program_groups(ctx, module)
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)
    sbt = create_sbt(program_grps)

    # Define rays: one hits, one misses
    ray_origins = cp.array([[0.25,0.25,1.0],
                            [2.0,2.0,1.0]], dtype=np.float32)
    ray_directions = cp.array([[0.0,0.0,-1.0],
                               [0.0,0.0,-1.0]], dtype=np.float32)

    hit_ids = ray_triangle_intersections(ctx, pipeline, sbt, gas,
                                         ray_origins, ray_directions)
    print("Hit IDs:", hit_ids.get())
