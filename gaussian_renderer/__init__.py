import torch
import torch.nn.functional as F

from gsplat.rendering import rasterization

from utils.sh_utils import eval_sh


def process_Gaussians(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    
    opacity = pc.get_opacity

    scales = pc.get_scaling #pc.get_scaling_with_3D_filter
    
    rotations = pc.rotation_activation(pc.splats["quats"])
    
    return means3D, rotations, opacity, colors, scales

def process_Gaussians_triplane(pc):
    # Use existing function for processing canon
    means3D, rotations, opacity, colors, scales = process_Gaussians(pc)
    
    invariance = pc.get_lambda
    texsample = pc.get_ab
    texscale = pc.get_texscale
        
    return means3D, rotations, opacity, colors, scales, texsample, texscale, invariance

def rendering_pass(means3D, rotation, scales, opacity, colors, invariance, cam, sh_deg=3, mode="RGB+D"):
    if mode in ['normals', '2D']:
        gmode = 'RGB+D'
    elif mode == "invariance":
        colors = invariance.unsqueeze(0)
        gmode = 'RGB'
        sh_deg=None
    else:
        gmode = mode
    
    try:
        intr = []
        w2c = []
        for c in cam:
            intr.append(c.intrinsics.unsqueeze(0))
            
            viewmat = c.w2c.unsqueeze(0)
            w2c.append(viewmat)
            
        intr = torch.cat(intr, dim=0)
        w2c = torch.cat(w2c, dim=0)
        width = c.image_width
        height = c.image_height
        if means3D.dim() == 3:
            intr = intr.unsqueeze(1)
            w2c = w2c.unsqueeze(1)
    except:
        intr = cam.intrinsics.unsqueeze(0)
        viewmat = cam.w2c.unsqueeze(0)
        w2c = viewmat
        width = cam.image_width
        height = cam.image_height
        
    # Typical RGB render of base color
    colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        w2c.cuda(), 
        intr.cuda(),
        width, 
        height,
        
        render_mode=gmode,
        
        # rasterize_mode='antialiased',
        # eps2d=0.3,
        
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        sh_degree=sh_deg, #pc.active_sh_degree,
    )
        
    return colors, alphas, meta
import torch.nn.functional as F

@torch.no_grad()
def make_cam_rays(camera):
    """Generate pixel-center camera rays in world space using the same intrinsics as gsplat."""
    H, W = camera.image_height, camera.image_width
    device = camera.device

    # Use the same intrinsics as gsplat (camera.K from getOptimalNewCameraMatrix)
    K = camera.intrinsics.to(device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Pixel-center coordinates
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    jj, ii = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    x = (ii - cx) / fx
    y = (jj - cy) / fy
    z = torch.ones_like(x)
    dirs_cam = torch.stack([x, y, z], dim=-1)  # [H,W,3]

    # We need the openGL c2w to align points rendered via openGL coordspace (even if we store our cameras as opencv)
    c2w = camera.pose.to(device)
    
    dirs_world = dirs_cam @ c2w[:3, :3].T
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
    origins_world = c2w[:3, 3].expand_as(dirs_world)

    return origins_world, dirs_world


def render_IBL_source(camera, abc, texture):
    H, W = camera.image_height, camera.image_width

    # 1. Camera rays
    origin, direction = make_cam_rays(camera)

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
    denom = (direction @ normal)
    valid = denom.abs() > 1e-8
    t = torch.empty_like(denom)
    t[valid] = (nom[valid] / denom[valid])
    x = origin + t * direction

    # Project onto u,v basis
    u_len2 = (u * u).sum()
    v_len2 = (v * v).sum()
    d = x - a
    u_coord = (d @ u) / u_len2   # [0,1] left→right
    v_coord = (d @ v) / v_len2   # [0,1] bottom→top
    
    # Only sample rays within patch
    mask = (u_coord >= 0) & (u_coord <= 1) & (v_coord >= 0) & (v_coord <= 1)
    mask = mask & (t.squeeze() > 0)
    # uv = torch.stack([u_coord, v_coord], dim=-1)  # (N,2)
    uv = torch.stack([u_coord, v_coord], dim=-1)
    uv = (2.0 * uv - 1.0).unsqueeze(0).unsqueeze(2)

    sampled = F.grid_sample(
        texture.unsqueeze(0), uv,
        align_corners=False, mode="bilinear"
    ).squeeze(0).squeeze(-1).reshape(3, H, W)
    
    mask = mask.reshape(H, W).float()
    sampled = sampled * mask.unsqueeze(0)

    return sampled

@torch.no_grad
def render(viewpoint_camera, pc, abc, texture, view_args=None, mip_level=2):
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
            means3D, rotation, opacity, colors, scales, texsample, texscale, invariance = process_Gaussians_triplane(pc)
            
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
        elif view_args['vis_mode'] == 'xyz':
            mean_max = means3D.max()
            mean_min = means3D.min()
            colors = (means3D - mean_min) / (mean_max - mean_min)
            colors = means3D.unsqueeze(0)
            
        elif view_args['vis_mode'] in ['deform', 'invariance']:
            colors = torch.cat([invariance, texsample], dim=-1)
            mode = "RGB"

        # Change for rendering with rgb instead of shs
        if view_args['vis_mode'] in ["xyz"]:
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
    
            if abc is not None:
                ibl = render_IBL_source(viewpoint_camera, abc, texture)
                ibl_alpha = ibl.mean(0).unsqueeze(0)

                alpha = alpha.squeeze(-1)
                render =  render * (alpha) + (1. - alpha) * ibl


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
            render = render[..., 0].repeat(3,1,1)
            
        elif view_args['vis_mode'] in 'deform':
            sample = render[..., 1:].squeeze(0).permute(2,0,1)
            render = sample_texture(texture.cuda(), sample)
        
        elif view_args['vis_mode'] == 'xyz':
            render = render.squeeze(0).permute(2,0,1)
    else:
        render, _ = render_extended([viewpoint_camera], pc, [texture], mip_level=mip_level)
        render = render.squeeze(0)
        
        
    return {
        "render": render,
        "extras":extras # A dict containing mor point info
        }

def render_extended(viewpoint_camera, pc, textures, return_canon=False, mip_level=2):
    """Fine/Deformation function

    Notes:
        Trains/Renders the deformed gaussians
            
    """
    # Sample triplanes and return Gaussian params + a,b,lambda
    means3D, rotation, opacity, colors_final, scales, texsample, texscale, invariance = process_Gaussians_triplane(pc)

    text_cols = torch.cat([invariance, texsample], dim=-1)    
    
    M = len(textures)
    means3D_final = means3D.unsqueeze(0).repeat(M, 1, 1)
    rotation_final = rotation.unsqueeze(0).repeat(M, 1, 1)
    scales_final = scales.unsqueeze(0).repeat(M, 1, 1)
    opacity_final = opacity.unsqueeze(0).repeat(M, 1, 1)
    colors_final = colors_final.unsqueeze(0).repeat(M, 1, 1, 1)
    text_cols = text_cols.unsqueeze(0).repeat(M, 1, 1, 1)

    colors_base, _, meta = rendering_pass(
        means3D_final,
        rotation_final, 
        scales_final, 
        opacity_final, 
        colors_final,
        None, 
        viewpoint_camera, 
        sh_deg=3,
        mode="RGB"
    )

    colors_base = colors_base.squeeze(1).permute(0, 3, 1, 2)
    
    colors_deform, _, _ = rendering_pass(
        means3D_final,
        rotation_final, 
        scales_final, 
        opacity_final, 
        text_cols,
        None, 
        viewpoint_camera, 
        sh_deg=3,
        mode="RGB"
    )

    colors_deform = colors_deform.squeeze(1).permute(0, 3, 1, 2)
    invariance_map = colors_deform[:, 0].unsqueeze(1)
    texsample_map = colors_deform[:, 1:]
    
    final_colors = []
    for i, (texture, col, inv, texsamp) in enumerate(zip(textures, colors_base, invariance_map, texsample_map)):
        # 3, H,W | 3,h,w | 1, H,W | 2, H, W ; h,w are for texture, H,W are for rendering
        
        delta_map = sample_texture(texture.cuda(), texsamp)
        final_color = col + inv*delta_map
        final_colors.append(final_color.unsqueeze(0))
        
    final_colors = torch.cat(final_colors, dim=0)
    if return_canon:
        return final_colors, colors_base, meta
    
    return final_colors, meta


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

    return colors.squeeze(0).permute(0, 3, 1, 2), meta

import torch.nn.functional as F
def sample_texture(I, uv):
    """
    args:
        uv, Tensor, 2, H,W
        I, Tensor, 3, h, w
    """
    # Normalize us -1, 1 (from 0, 1)
    uv = 2.*uv -1.
    uv = uv.permute(1,2,0).unsqueeze(0) # 1,H,W,2
    
    map = I.unsqueeze(0)
    mip_samples = F.grid_sample(map, uv, mode='bilinear', align_corners=False).squeeze(0)

    return mip_samples

    
