#pragma once

struct Params {
    float3* ray_origins;
    float3* ray_directions;
    int*    hit_ids;
    int     num_rays;
    OptixTraversableHandle handle;
};



struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};
