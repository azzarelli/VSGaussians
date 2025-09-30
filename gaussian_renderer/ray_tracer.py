import os
import optix as ox
import cupy as cp
import numpy as np

DEBUG = True

if DEBUG:
    exception_flags = ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW
    debug_level = ox.CompileDebugLevel.FULL
    opt_level = ox.CompileOptimizationLevel.LEVEL_0
else:
    exception_flags = ox.ExceptionFlags.NONE
    debug_level = ox.CompileDebugLevel.NONE
    opt_level = ox.CompileOptimizationLevel.LEVEL_3


def create_module(ctx, pipeline_opts, cuda_src):
    compile_opts = ox.ModuleCompileOptions(
        debug_level=debug_level,
        opt_level=opt_level
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
                                       debug_level=debug_level)

    pipeline=  ox.Pipeline(ctx,
                           compile_options=pipeline_options,
                           link_options=link_opts,
                           program_groups=program_grps)
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                0,  # max_cc_depth
                                1)  # max_dc_depth
    return pipeline

def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp, names=('rgb',), formats=('3f4',))
    miss_sbt['rgb'] = [0., 0., 0.]

    hit_sbt = ox.SbtRecord(hit_grp)
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

    return sbt

def log_callback(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))
    


    
class OptixTriangles:
    
    def __init__(self):
        ctx = ox.DeviceContext(
            validation_mode=False,
            log_callback_function=log_callback,
            log_callback_level=3
        )

        script_dir = "./submodules/python-optix/examples/"
        cuda_src = os.path.join(script_dir, "cuda", "triangle_ray.cu")
        pipeline_options = ox.PipelineCompileOptions(
        traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
        num_payload_values=3,
        num_attribute_values=3,
        exception_flags=ox.ExceptionFlags.NONE,
        pipeline_launch_params_variable_name="params"
        )
        self.module = create_module(ctx, pipeline_options, cuda_src)
        self.program_grps = create_program_groups(ctx, self.module)
        self.pipeline = create_pipeline(ctx, self.program_grps, pipeline_options)
        self.sbt = create_sbt(self.program_grps)
        
        
        self.ctx = ctx
        self.pipeline_options = pipeline_options

        self.vertices = None
        self.build_input = None
        self.gas = None
        
    def update_gas(self, vertices):
        """Build Geom Accell Structure from triangles and store data
        """
        self.vertices = vertices

        build_input = ox.BuildInputTriangleArray(vertices, flags=[ox.GeometryFlags.NONE])
        self.gas = ox.AccelerationStructure(
            self.ctx, 
            build_input,
            # allow_update=True, 
            compact=True,
            # random_vertex_access=True
        )
        
    def trace(self, cam):
        H,W = cam.image_height, cam.image_width

        # Build launch params
        params_tmp = [
            ('u8',  'image'),
            ('u4',  'image_width'),
            ('u4',  'image_height'),
            ('3f4', 'cam_eye'),
            ('3f4', 'cam_u'),
            ('3f4', 'cam_v'),
            ('3f4', 'cam_w'),
            ('u8',  'trav_handle')
        ]

        params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                    formats=[p[0] for p in params_tmp])
        c2w = cam.pose.cpu().numpy()
        R = c2w[:3,:3]
        t = c2w[:3,3]


        # params['cam_eye'] = t
        # params['cam_U']   = R[:,0]   # right
        # params['cam_V']   = R[:,1]   # up
        # params['cam_W']   = R[:,2]   # forward


        output_image = np.zeros((H,W, 4), 'B')
        output_image[:, :, :] = [255, 128, 0, 255]
        output_image = cp.asarray(output_image)

        params['image'] = output_image.data.ptr
        params['image_width'] = W
        params['image_height'] =  H
        c2w = cam.pose.cpu().numpy()
        R = c2w[:3,:3]
        t = c2w[:3,3]

        W = cam.image_width
        H = cam.image_height
        fx, fy = cam.fx, cam.fy

        # Scale factors from NDC [-1,1] to pixels
        scale_u = W / (2.0 * fx)
        scale_v = H / (2.0 * fy)

        # Right, Up, Forward vectors from pose
        right   = R[:,0]
        up      = R[:,1]
        forward = -R[:,2]   # because your raygen assumes looking down -Z

        params['cam_eye'] = t
        params['cam_u']   = right   * scale_u
        params['cam_v']   = up      * scale_v
        params['cam_w']   = -forward

        params['trav_handle'] = self.gas.handle

        stream = cp.cuda.Stream()

        self.pipeline.launch(self.sbt, dimensions=(W,H), params=params, stream=stream)

        stream.synchronize()

        return output_image #cp.asnumpy(output_image)
       
    