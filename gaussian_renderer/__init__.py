import torch
import math
import torch.nn.functional as F
import cupy as cp

from gsplat.rendering import rasterization, rasterization_2dgs

from gaussian_renderer.ray_tracer import RaycastSTE

def render_IBL_source(cam, abc, texture, device="cuda"):
    intr = cam.intrinsics
    fx,fy,cx,cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    origin, direction = cam.generate_rays(None, None, None, fx,fy,cx,cy)
    
    return cam.surface_sample(origin, direction, abc, texture.cuda())


def process_Gaussians(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.splats["quats"]
    

    rotations = pc.rotation_activation(rotations)
    
        
    return means3D, rotations, opacity, colors, scales

def process_overfit_Gaussians(pc, time):
    means3D = pc.get_xyz
    colors = pc.get_features
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.splats["quats"]
    
    invariance = pc.get_invariance_coefficient    

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
 
 
def process_Gaussians_triplane(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.splats["quats"]
    
    params, invariance = pc._deformation(
        point=means3D,
    )
    
    rotations = pc.rotation_activation(rotations)
    
    return means3D, rotations, opacity, colors, scales, params, invariance

 
def rendering_pass(means3D, rotation, scales, opacity, colors, invariance, cam, sh_deg=3, mode="RGB+D"):
    if mode in ['normals', '2D']:
        gmode = 'RGB+D'
    elif mode == "invariance":
        colors = invariance.unsqueeze(0)
        gmode = 'RGB'
        sh_deg=None
    else:
        gmode = mode
    
    # Typical RGB render of base color
    colors, alphas, meta = rasterization(
    # colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        cam.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
        cam.intrinsics.unsqueeze(0).cuda(),
        cam.image_width, 
        cam.image_height,
        
        render_mode=gmode,
        # rasterize_mode='antialiased',
        # eps2d=0.1,
        sh_degree=sh_deg, #pc.active_sh_degree,
        packed=True
    )
    
    colors = colors[..., :3]
    
    # if mode == 'normals':
    #     colors = surf_normals
    # elif mode == '2D':
    #     colors = median_depth
        
    return colors, alphas, meta #(normals, surf_normals, distort, median_depth, meta)


def sample_IBL(origin, direction, abc, texture):
    H, W = origin.shape[:2]
    origin = origin.reshape(-1, 3)
    direction = direction.reshape(-1, 3)
    
    a,b,c = abc
    # Calculate t intersection for each ray
    u = b-a
    v = c-a
    normal = torch.cross(u, v, dim=-1)
    normal = F.normalize(normal, dim=-1).unsqueeze(-1)

    # Intersect rays with plane
    nom = (a.unsqueeze(0) - origin) @ normal
    denom = (direction @ normal).clamp(min=1e-8)
    t = nom / denom
    x = origin + t * direction

    # Project onto u,v basis
    u_len2 = (u * u).sum()
    v_len2 = (v * v).sum()
    d = x - a
    u_coord = (d @ u) / u_len2   # [0,1] left→right
    v_coord = (d @ v) / v_len2   # [0,1] bottom→top

    # grid_sample expects (x,y) with y=down, so we flip v
    v_coord = 1. - v_coord

    hit_mask = (t.squeeze(-1) > 0) & \
        (u_coord >= 0) & (u_coord <= 1) & \
        (v_coord >= 0) & (v_coord <= 1)
               
    uv = torch.stack([u_coord, v_coord], dim=-1)  # (N,2)
    uv = (2.0 * uv - 1.0).unsqueeze(0).unsqueeze(2)

    sampled = F.grid_sample(
        texture.unsqueeze(0), uv,
        align_corners=True, mode="bilinear"
    ).squeeze(0).squeeze(-1).reshape(3, H, W)
    
    hit_mask = hit_mask.reshape(H, W)
    hit_indices = torch.nonzero(hit_mask, as_tuple=False)

    return sampled, hit_mask, hit_indices

SIGNS = torch.tensor([[1, 1],
                      [-1, 1],
                      [1, -1],
                      [-1, -1]], dtype=torch.float).cuda()

def generate_triangles(means, mag, dirs, colors, opacity, thresh=0.1):
     # Determine the point in local Gaussian space where the curve of the 2-D Gaussian in the top-right (positive x-y) quadrant
    #  reached y=-x
    # c = 3.218875825 # fixed confidence at 0.8 
    # mag_sq = mag **2
    # a_prime = mag_sq[:, 0] * torch.sqrt(c/(mag_sq[:, 0] + mag_sq[:, 1]))  
    # b_prime = mag_sq[:, 1] * torch.sqrt(c/(mag_sq[:, 0] + mag_sq[:, 1]))  
    # print(a_prime.shape, mag.shape)
    # exit()
    
    signs = torch.tensor([[1, 1],
                      [-1, 1],
                      [1, -1],
                      [-1, -1]], device=mag.device, dtype=mag.dtype)
    
    mask = (opacity.squeeze(-1) > thresh)
    colors = colors[mask]
    means = means[mask]
    mag = mag[mask]
    dirs = dirs[mask]
    
    # Expand for broadcasting: (N, 4, 2, 3)
    reflected = means.unsqueeze(1).unsqueeze(1) + (signs.unsqueeze(0).unsqueeze(-1) * mag.unsqueeze(1).unsqueeze(-1) * dirs.unsqueeze(1))

    # reflected: (N, 4, 2, 3)
    # Concatenate the center point to each reflected triangle
    tris = torch.cat([means.unsqueeze(1).unsqueeze(1).expand(-1, 4, -1, -1), reflected], dim=2)  # (N, 4, 3, 3)

    # Flatten all triangles
    return tris.reshape(-1, 3), colors

def r_render(x, d, n, abc, texture, optix_runner, means3D, pc, colors, opacity, N=4, update_verts=False):
    """Ray-Screen Intersection w/ intersection mask
        args:
            N : int, number of triangles per gaussian
            update_verts : bool, one needs to be here once
    """
    x,d = x.squeeze(0), d.squeeze(0)
    render, hit_mask, hit_indices = sample_IBL(x, d, abc, texture.cuda())
    # Select x and d for sampling scene, no need to resample scene where ray-screen intersection already exists
    # x_, d_ = x[~hit_mask], d[~hit_mask]
    
    """Ray-Scene Intersection w/ masked origin and directions
    """
    # Generate triangle primitives from Gaussian
    if update_verts:
        mag, dirs = pc.get_covmat
        verts, colors = generate_triangles(means3D, mag, dirs, colors, opacity)
        verts = verts.detach()
    else:
        verts = None
    # Forward through runner
    buffer_image = RaycastSTE.apply(x,d,N,colors, verts, optix_runner, update_verts)
    

    render = render.permute(1,2,0)
    render[~hit_mask] = buffer_image[~hit_mask]
    
    return render

@torch.no_grad
def render(viewpoint_camera, pc, abc, texture, view_args=None):
    """
    Render the scene for viewing
    """
    extras = None

    time = torch.tensor(viewpoint_camera.time).to(pc.splats["means"].device).repeat(pc.splats["means"].shape[0], 1)
    
    if view_args["finecoarse_flag"]:
        if view_args['vis_mode'] not in ['invariance', 'deform']:
            means3D, rotation, opacity, colors, scales = process_Gaussians(pc)
            invariance = None
        else:
            means3D, rotation, opacity, colors, scales, params, invariance = process_Gaussians_triplane(pc)

        if view_args["stage"] == "ba":
            scales *= 0.005
        
        active_sh = pc.active_sh_degree
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
        elif view_args['vis_mode'] == 'deform':
            colors = sample_mipmap(texture, params[:, :-1], params[:, -1].unsqueeze(-1), num_levels=3).unsqueeze(0)
            mode = "RGB"
            active_sh=None
        
        # Render
        render, alpha, _ = rendering_pass(
            means3D, rotation, scales, opacity, colors, invariance,
            viewpoint_camera, 
            active_sh,
            mode=mode
        )
        
        if view_args['vis_mode'] == 'normals' or view_args['vis_mode'] == '2D':
            view_args['vis_mode'] = 'render'
        
        # Process image
        if view_args['vis_mode'] == 'render':
            render = render.squeeze(0).permute(2,0,1)
            
            # ibl = render_IBL_source(viewpoint_camera, abc, texture)
            # alpha = alpha.squeeze(-1)
            # render = render * (alpha) + (1-alpha) * ibl
            
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
            
        elif view_args['vis_mode'] == 'invariance':
            render = render.squeeze(-1).repeat(3,1,1)
        elif view_args['vis_mode'] in 'deform':
            render = render.squeeze(0).permute(2,0,1)
        
    else:
        render, _ = render_extended(viewpoint_camera, pc, texture)

    return {
        "render": render,
        "extras":extras # A dict containing mor point info
        }

from utils.sh_utils import SH2RGB


import time as TIME
@torch.no_grad 
def render_triangles(viewpoint_camera, pc, optix_runner):
    """
    Render the scene for viewing
    """
    time = torch.tensor(viewpoint_camera.time).to(pc.splats["means"].device).repeat(pc.splats["means"].shape[0], 1)
    means3D, rotation, opacity, colors, scales = process_Gaussians(pc)
    
    x, d = viewpoint_camera.generate_rays(ctype="tris")

    mag, dirs = pc.get_covmat
    verts, colors = generate_triangles(means3D, mag, dirs, colors, opacity)
    colors_rgb = SH2RGB(colors[:, :3]).clamp(0.0, 1.0)

    verts = verts.detach()
    N = 4
    # Forward through runner
    buffer_image = RaycastSTE.apply(x, d, N, colors_rgb, verts, optix_runner, False)
    

    # buffer_image[buffer_hitIndices > 0] *= 0.
    # buffer_image[buffer_hitIndices > 0] += 1.
    
    return buffer_image


from utils.sh_utils import eval_sh
def render_extended(viewpoint_camera, pc, texture, return_canon=False):
    """Fine/Deformation function

    Notes:
        Trains/Renders the deformed gaussians
            
    """
    # Sample triplanes and return Gaussian params + a,b,lambda
    means3D, rotation, opacity, colors, scales, params, invariance = process_Gaussians_triplane(pc)
    
    # Precompute point colors
    shs_view = colors.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(colors.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
    # Precompute point a,b,s texture indexing

    colors_ibl = sample_mipmap(texture, params[:, :-1], params[:, -1].unsqueeze(-1), num_levels=3)
    colors_ = invariance*colors_precomp + (1.-invariance)*colors_ibl

    
    colors_deform, _, meta = rendering_pass(
        means3D, 
        rotation, 
        scales, 
        opacity, 
        colors_.unsqueeze(0),
        None, 
        viewpoint_camera, 
        None,
        mode="RGB"
    )

    if return_canon:
        colors_canon, _, meta = rendering_pass(
            means3D, 
            rotation, 
            scales, 
            opacity, 
            colors_precomp.unsqueeze(0), 
            None, 
            viewpoint_camera, 
            None,
            mode="RGB"
        )
        return colors_deform.squeeze(0).permute(2,0,1), colors_canon.squeeze(0).permute(2,0,1), meta
    return colors_deform.squeeze(0).permute(2,0,1), meta


def render_canonical(viewpoint_camera, pc):
    """Canonical rendering function

    """
    means3D, rotation, opacity, colors, scales = process_Gaussians(pc)
    
    colors, _, meta = rendering_pass(
        means3D, 
        rotation, 
        scales, 
        opacity, 
        colors,
        None, 
        viewpoint_camera, 
        pc.active_sh_degree,
        mode="RGB"
    )
    
    return colors.squeeze(0).permute(2,0,1), meta

import torch.nn.functional as F
def generate_mipmaps(I, num_levels=3):
    I = I.unsqueeze(0)
    maps = [I]
    _, C,H,W = I.shape
    
    for _ in range(1, num_levels):
        I = F.interpolate(
            I, scale_factor=0.5,
            mode='bilinear', align_corners=False,
            recompute_scale_factor=True
        )
        maps.append(I)
        
    return maps

def sample_mipmap(I, uv, s, num_levels=3):
    """
    args:
        uv, Tensor, N,2
        s, Tensor, N,1
        I, Tensor, 3, H, W
    """
    N = s.size(0)
    # 1. Generate mipmaps
    maps = generate_mipmaps(I, num_levels=num_levels) # shaped list: 1,3,h,w where h,w, are the downsampled height and width per level
    
    # Normalize us -1, 1 (from 0, 1)
    uv = 2.*uv -1.
    uv = uv.unsqueeze(0).unsqueeze(0) # for grid_sample input we need, N,Hout,Wout,2, where N =1, and W=number of points
    
    # Scaling mip-maps
    L = s*(num_levels-1.)
    lower = torch.floor(L).long().clamp(max=num_levels-1)
    upper = torch.clamp(lower + 1, max=num_levels-1)
    s_interp = (L - lower.float())
    
    # Initialize mipmap samples
    mip_samples = torch.empty((N, num_levels, 3), device=s.device)    
    # For each map sample using u,v and store the values in samples
    for idx, map in enumerate(maps):
        # map is (1, 3, h, w)
        
        mip_samples[:, idx] = F.grid_sample(map, uv, mode='bilinear', align_corners=False).squeeze(2).squeeze(0).permute(1,0)
    
    gather_idx_low  = lower.view(N, 1, 1).expand(-1, 1, 3)
    gather_idx_high = upper.view(N, 1, 1).expand(-1, 1, 3)

    colors_low  = torch.gather(mip_samples, 1, gather_idx_low).squeeze(1)   # [N,3]
    colors_high = torch.gather(mip_samples, 1, gather_idx_high).squeeze(1) 

    colors = (1. - s_interp) * colors_low + s_interp * colors_high
    
    return colors
    