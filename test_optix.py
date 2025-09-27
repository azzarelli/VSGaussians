import torch
import optix as ox
from pathlib import Path

ctx = ox.DeviceContext()

ptx_path = "submodules/python-optix/shaders/minimal.ptx"
module = ox.Module(ctx, ptx_path)

ctx = ox.DeviceContext()

ptx_path = Path("submodules/python-optix/shaders/minimal.ptx")
module = ox.Module(ctx, str(ptx_path))

raygen_prog = ox.ProgramGroup(module, "__raygen__rg")
hit_prog    = ox.ProgramGroup(module, "__closesthit__ch")
miss_prog   = ox.ProgramGroup(module, "__miss__ms")

pipeline = ox.Pipeline(ctx, [raygen_prog, hit_prog, miss_prog])
sbt = ox.ShaderBindingTable(raygen=raygen_prog, miss=miss_prog, hit=hit_prog)


# --- 2. Define the function ---
def ray_triangle_intersections(ray_origins, ray_directions, triangles):
    """
    ray_origins: (N,3) torch tensor (float32, cuda)
    ray_directions: (N,3) torch tensor (float32, cuda)
    triangles: (M,3,3) torch tensor (float32, cuda)
    Returns: (N,1) torch tensor with triangle IDs (or -1 if miss)
    """
    N = ray_origins.shape[0]

    ray_origins   = ray_origins.contiguous()
    ray_directions = ray_directions.contiguous()
    triangles      = triangles.contiguous()

    hit_ids = torch.full((N,1), -1, dtype=torch.int32, device="cuda")

    # Build geometry acceleration structure (GAS)
    tri_array = ox.GeometryTriangles(vertices=triangles.data_ptr(),
                                     num_triangles=triangles.shape[0])
    gas = ox.build_acceleration_structure(ctx, [tri_array])

    # Launch pipeline
    pipeline.launch(
        sbt,
        dimensions=(N,1,1),
        params={
            "ray_origins":   ray_origins.data_ptr(),
            "ray_directions": ray_directions.data_ptr(),
            "hit_ids":       hit_ids.data_ptr(),
            "num_rays":      N,
            "gas_handle":    gas.handle,
        }
    )

    return hit_ids


# --- 3. Quick test ---
if __name__ == "__main__":
    # One triangle in xy-plane
    triangles = torch.tensor([[[0,0,0],[1,0,0],[0,1,0]]],
                             dtype=torch.float32, device="cuda")

    # Rays
    ray_origins = torch.tensor([[0.25, 0.25, 1.0],
                                [2.0, 2.0, 1.0]], dtype=torch.float32, device="cuda")
    ray_directions = torch.tensor([[0,0,-1],
                                   [0,0,-1]], dtype=torch.float32, device="cuda")

    hits = ray_triangle_intersections(ray_origins, ray_directions, triangles)
    print("Hit IDs:", hits.cpu().numpy())
