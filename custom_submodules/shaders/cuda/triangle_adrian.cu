#include <optix.h>

#include "triangle_adrian.h"
#include "helpers.h"

#include "vec_math.h"

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}





extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const int ray_id = idx.x;
    if (ray_id >= params.num_rays) return;

    float3 ray_o = params.ray_origins[ray_id];
    float3 ray_d = params.ray_directions[ray_id];

    unsigned int hit_id = 0;
    optixTrace(
        params.handle,
        ray_o, ray_d,
        0.0f, 1e16f, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        hit_id
    );

    params.hit_ids[ray_id] = static_cast<int>(hit_id);
}


extern "C" __global__ void __closesthit__ch() {
    unsigned int prim_idx = optixGetPrimitiveIndex();
    optixSetPayload_0(prim_idx);
}

extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(static_cast<unsigned int>(-1));
}
