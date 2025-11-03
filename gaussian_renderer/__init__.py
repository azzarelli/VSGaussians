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


# @torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

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
            c2w = c.pose
            
            viewmat = get_viewmat(c2w.unsqueeze(0))
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
        c2w = cam.pose
        viewmat = get_viewmat(c2w.unsqueeze(0))
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
            means3D, rotation, opacity, colors, scales, texsample, texscale, invariance = process_Gaussians_triplane(pc)
            
            shs_view = texsample.transpose(1, 2).view(-1, 2, 16)
            dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(texsample.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            texsample_ab = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
            shs_view = invariance.transpose(1, 2).view(-1, 1, 16)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            invariance = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
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
            colors = sample_mipmap(texture, texsample_ab, texscale, num_levels=2).unsqueeze(0)
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
        render, _ = render_extended([viewpoint_camera], pc, [texture])
        render = render.squeeze(0)
    return {
        "render": render,
        "extras":extras # A dict containing mor point info
        }

def render_extended(viewpoint_camera, pc, textures, return_canon=False):
    """Fine/Deformation function

    Notes:
        Trains/Renders the deformed gaussians
            
    """
    # Sample triplanes and return Gaussian params + a,b,lambda
    means3D, rotation, opacity, colors, scales, texsample, texscale, invariance = process_Gaussians_triplane(pc)
    
    # Precompute point a,b,s texture indexing
    colors_final = []
    for texture, cam in zip(textures, viewpoint_camera):
        dir_pp = (means3D - cam.camera_center.cuda().repeat(colors.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        shs_view = colors.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        shs_view = texsample.transpose(1, 2).view(-1, 2, 16)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        texsample_ab = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        shs_view = invariance.transpose(1, 2).view(-1, 1, 16)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        tex_invariance = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        colors_ibl = sample_mipmap(texture.cuda(), texsample_ab, texscale, num_levels=2)
        color_d = tex_invariance*colors_ibl
        colors_final.append((colors_precomp + color_d).unsqueeze(0))
        
    colors_final = torch.cat(colors_final, dim=0)
    M = len(textures)
    means3D_final = means3D.unsqueeze(0).repeat(M, 1, 1)
    rotation_final = rotation.unsqueeze(0).repeat(M, 1, 1)
    scales_final = scales.unsqueeze(0).repeat(M, 1, 1)
    opacity_final = opacity.unsqueeze(0).repeat(M, 1, 1)

    colors_deform, _, meta = rendering_pass(
        means3D_final,
        rotation_final, 
        scales_final, 
        opacity_final, 
        colors_final,
        None, 
        viewpoint_camera, 
        None,
        mode="RGB"
    )

    colors_deform = colors_deform.squeeze(1).permute(0, 3, 1, 2)
    
    if return_canon:
        colors_canon, _, _ = rendering_pass(
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
        colors_canon = colors_canon.squeeze(1).permute(0, 3, 1, 2)

        return colors_deform, colors_canon, meta
    return colors_deform, meta


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
    maps = I.unsqueeze(0) # F.interpolate(
    #     I.unsqueeze(0), scale_factor=0.5,
    #     mode='bilinear', align_corners=False,
    #     recompute_scale_factor=True
    # ) # shaped list: 1,3,h,w where h,w, are the downsampled height and width per level
    
    # Normalize us -1, 1 (from 0, 1)
    uv = 2.*uv -1.
    uv = uv.unsqueeze(0).unsqueeze(0) # for grid_sample input we need, N,Hout,Wout,2, where N =1, and W=number of points
    
    # Scaling mip-maps

    mip_samples = F.grid_sample(maps, uv, mode='bilinear', align_corners=False).squeeze(2).squeeze(0).permute(1,0)
    return mip_samples
    