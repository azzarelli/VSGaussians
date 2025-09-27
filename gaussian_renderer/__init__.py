import torch
import math
import torch.nn.functional as F

from gsplat.rendering import rasterization, rasterization_2dgs



def render_IBL_source(cam, abc, texture, device="cuda"):
    intr = cam.intrinsics
    fx,fy,cx,cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    origin, direction = cam.generate_rays(None, None, None, fx,fy,cx,cy)
    
    return cam.surface_sample(origin, direction, abc, texture.cuda())


def process_Gaussians(pc, time):
    means3D = pc.get_xyz
    colors = pc.get_features
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.splats["quats"]
    
    invariance = pc.get_invariance    

    means3D, rotations, opacity, colors, extras = pc._deformation(
        point=means3D, 
        rotations=rotations,
        scales=scales,
        times_sel=time,
        invariance=invariance,
        h_emb=opacity,
        shs=colors,
    )

    rotations = pc.rotation_activation(rotations)
    
        
    return means3D, rotations, opacity, colors, scales, invariance
 
def rendering_pass(means3D, rotation, scales, opacity, colors, invariance, cam, sh_deg=3, mode="RGB"):
    if mode in ['normals', '2D']:
        gmode = 'RGB'
    elif mode == "invariance":
        colors = invariance.unsqueeze(0)
        opacity = opacity.clone().detach() * 1.0
        gmode = 'RGB'
        sh_deg=None
    else:
        gmode = mode
    
    # Typical RGB render of base color
    colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
    # colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        cam.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
        cam.intrinsics.unsqueeze(0).cuda(),
        cam.image_width, 
        cam.image_height,
        
        render_mode=gmode,
        # rasterize_mode='antialiased',
        # eps2d=0.1,
        sh_degree=sh_deg #pc.active_sh_degree
    )
    
    if mode == 'normals':
        colors = normals
    elif mode == '2D':
        colors = median_depth
        
    return colors, alphas, (normals, surf_normals, distort, median_depth, meta)

@torch.no_grad
def render(viewpoint_camera, pc, abc, texture, view_args=None):
    """
    Render the scene for viewing
    """
    extras = None

    time = torch.tensor(viewpoint_camera.time).to(pc.splats["means"].device).repeat(pc.splats["means"].shape[0], 1)
    means3D, rotation, opacity, colors, scales, invariance = process_Gaussians(pc, time)
    if view_args["stage"] == "ba":
        scales *= 0.005

    # Set arguments depending on type of viewing
    if view_args['vis_mode'] in 'render':
        mode = "RGB"
    elif view_args['vis_mode'] == 'alpha':
        mode = "RGB"
    elif view_args['vis_mode'] == 'normals':
        mode = "normals"
    elif view_args['vis_mode'] == '2D':
        mode = "2D"
    elif view_args['vis_mode'] == 'D':
        mode = "D"
    elif view_args['vis_mode'] == 'ED':
        mode = "ED"
    elif view_args['vis_mode'] == 'invariance':
        mode = "invariance"
    
    # Render
    render, alpha, _ = rendering_pass(
        means3D, rotation, scales, opacity, colors, invariance,
        viewpoint_camera, 
        pc.active_sh_degree,
        mode=mode
    )
    
    if view_args['vis_mode'] == 'normals' or view_args['vis_mode'] == '2D':
        view_args['vis_mode'] = 'render'
    
    # Process image
    if view_args['vis_mode'] in 'render':
        render = render.squeeze(0).permute(2,0,1)
        
        ibl = render_IBL_source(viewpoint_camera, abc, texture)
        alpha = alpha.squeeze(-1)
        render = render * (alpha) + (1-alpha) * ibl
        
    elif view_args['vis_mode'] == 'alpha':
        render = alpha
        render = (render - render.min())/ (render.max() - render.min())
        render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'D':
        render = (render - render.min())/ (render.max() - render.min())
        render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
        
    elif view_args['vis_mode'] == 'ED':
        render = (render - render.min())/ (render.max() - render.min())
        render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
    
    return {
        "render": render,
        "extras":extras # A dict containing mor point info
        }


def ray_trace(verts, faces, origins, directions):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # Calculate the edge vectors of the triangle plane
    e1 = v1 - v0
    e2 = v2 - v0
    
    HW = origins.shape[:1]
    origins = origins.view(-1, 3)
    directions = directions.view(-1, 3)
    
    print(origins.shape, directions.shape)
    RD = directions[:, None, :]
    RO = origins[:, None, :]
    
    V0 = v0[None, :, :]
    E1 = e1[None, :, :]
    E2 = e2[None, :, :]
    
    F = RD.shape[0]
    F_chunk = 50000
    for tri_start in range(0, F, F_chunk):
        tri_end = min(tri_start + F_chunk, F)
        Rd = RD[tri_start:tri_end]
        Ro = RO[tri_start:tri_end]
        print(Rd.shape)
        pvec = torch.cross(Rd, E2, dim=-1)
        det  = (E1 * pvec).sum(-1)  
    
    print(Rd.shape, E2.shape)
    exit()
    
    
    exit()

@torch.no_grad 
def render_triangles(viewpoint_camera, pc, abc, texture):
    """
    Render the scene for viewing
    """
    extras = None

    time = torch.tensor(viewpoint_camera.time).to(pc.splats["means"].device).repeat(pc.splats["means"].shape[0], 1)
    means3D, rotation, opacity, colors, scales, invariance = process_Gaussians(pc, time)
    means = means3D.unsqueeze(1)
    mag, dirs = pc.get_covmat
    # verts = torch.cat([means, tangents], dim=1)
    
    # Determine the point in local Gaussian space where the curve of the 2-D Gaussian in the top-right (positive x-y) quadrant
    #  reached y=-x
    # c = 3.218875825 # fixed confidence at 0.8 
    # mag_sq = mag **2
    # a_prime = mag_sq[:, 0] * torch.sqrt(c/(mag_sq[:, 0] + mag_sq[:, 1]))  
    # b_prime = mag_sq[:, 1] * torch.sqrt(c/(mag_sq[:, 0] + mag_sq[:, 1]))  
    # print(a_prime.shape, mag.shape)
    # exit()
    tris = means3D.unsqueeze(1) + mag.unsqueeze(-1)*dirs
    tris = torch.cat([means3D.unsqueeze(1), tris], dim=1) # N,3,3, where vertices are centre, top and right.
    verts = tris.reshape(-1, 3)
    faces = torch.arange(tris.shape[0] * 3, device=tris.device).reshape(tris.shape[0], 3)
    rays_o, rays_d = viewpoint_camera.generate_rays()

    out = ray_trace(verts, faces, rays_o, rays_d)
    
    # out = kln.prepare_vertices(
    #     verts, faces, 
    # )
    # hit, face_idx, bary_coords, depth = ray_intersect(
    #     verts, faces, rays_o, rays_d
    # )
    print(dir(kln))
    
    # print(hit)
    # print(means.shape)
    exit()

    
    # Render
    render, alpha, _ = rendering_pass(
        means3D, rotation, scales, opacity, colors, invariance,
        viewpoint_camera, 
        pc.active_sh_degree,
        mode='RGB'
    )
    

    render = render.squeeze(0).permute(2,0,1)
    
    ibl = render_IBL_source(viewpoint_camera, abc, texture)
    alpha = alpha.squeeze(-1)
    render = render * (alpha) + (1-alpha) * ibl

    return render



def render_batch(viewpoint_camera, pc):
    """Training renderer the scene
    
        single bactch size
    """
    time = torch.tensor(viewpoint_camera.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1).detach()
    time = time*0. +viewpoint_camera.time
    
    means3D, rotation, opacity, colors, scales, invariance = process_Gaussians(pc, time)
    
    # Set up rasterization configuration
    rgb, _, meta = rendering_pass(means3D, rotation, scales, opacity, colors, invariance, viewpoint_camera, pc.active_sh_degree)
    rgb = rgb.squeeze(0).permute(2,0,1)
    
    return rgb, meta[-1]

