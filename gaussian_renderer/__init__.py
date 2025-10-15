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
    
    invariance = pc.get_invariance_coefficient    

    rotations = pc.rotation_activation(rotations)
    
        
    return means3D, rotations, opacity, colors, scales, invariance

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
 

def process_Gaussians_extended(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.splats["quats"]
    
    invariance = pc.get_invariance_coefficient    
    reflections = pc.get_reflection_coefficient    
    refraction = pc.get_refraction_coefficient    

    rotations = pc.rotation_activation(rotations)
        
    return means3D, rotations, opacity, colors, scales, invariance, reflections, refraction
 
def rendering_pass(means3D, rotation, scales, opacity, colors, invariance, cam, sh_deg=3, mode="RGB+D"):
    if mode in ['normals', '2D']:
        gmode = 'RGB+D'
    elif mode == "invariance":
        colors = invariance.unsqueeze(0)
        opacity = opacity.detach()
        means3D = means3D.detach()
        rotation = rotation.detach()
        scales = scales.detach()
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
    
    colors = colors[..., :3]
    
    if mode == 'normals':
        colors = surf_normals
    elif mode == '2D':
        colors = median_depth
        
    return colors, alphas, (normals, surf_normals, distort, median_depth, meta)

def rendering_base_intensity(means3D, rotation, scales,opacity, color, cam, sh_deg=3):
    colors = color.unsqueeze(0)
    opacity = opacity.clone().detach()*0. + 1.0
    gmode = 'RGB'
    sh_deg=None

    # Typical RGB render of base color
    colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
    # colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        cam.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
        cam.intrinsics.unsqueeze(0).cuda(),
        cam.image_width, 
        cam.image_height,
        
        render_mode=gmode,
        sh_degree=sh_deg
    )
        
    return colors, alphas


def render_multi_feature(means3D, rotation, scales, opacity, cam, sh_deg=3):
    # Here we want to render the position
    # min_x = means3D[:, 0].min()
    # denom_x = means3D[:, 0].max() - min_x
    # min_y = means3D[:, 1].min()
    # denom_y = means3D[:, 1].max() - min_y
    # min_z = means3D[:, 2].min()
    # denom_z = means3D[:, 2].max() - min_z
    
    # # Scale
    # norm_means = means3D.clone()
    # norm_means[:, 0] = (means3D[:, 0] - min_x)/denom_x
    # norm_means[:, 1] = (means3D[:, 1] - min_y)/denom_y
    # norm_means[:, 2] = (means3D[:, 2] - min_z)/denom_z
    
    colors = means3D.unsqueeze(0)
    opacity = opacity.clone().detach()*0. + 1.0
    gmode = 'RGB'
    sh_deg=None

    # Typical RGB render of base color
    xyzs, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
    # colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        cam.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
        cam.intrinsics.unsqueeze(0).cuda(),
        cam.image_width, 
        cam.image_height,
        
        render_mode=gmode,
        sh_degree=sh_deg
    )
    # Unscale
    # xyzs[..., 0] = (xyzs[..., 0] * denom_x) + min_x
    # xyzs[..., 1] = (xyzs[..., 1] * denom_y) + min_y
    # xyzs[..., 2] = (xyzs[..., 2] * denom_z) + min_z

    _, direction = cam.generate_rays(ctype="tris")
    
    return xyzs, direction, normals.squeeze(0)


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
        means3D, rotation, opacity, colors, scales, invariance = process_Gaussians(pc)
    else:
        means3D, rotation, opacity, colors, scales, invariance = process_overfit_Gaussians(pc, time)
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

from utils.sh_utils import SH2RGB


import time as TIME
@torch.no_grad 
def render_triangles(viewpoint_camera, pc, optix_runner):
    """
    Render the scene for viewing
    """
    time = torch.tensor(viewpoint_camera.time).to(pc.splats["means"].device).repeat(pc.splats["means"].shape[0], 1)
    means3D, rotation, opacity, colors, scales, invariance = process_Gaussians(pc)
    
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


def render_batch(viewpoint_camera, pc):
    """Training renderer the scene
    
        single bactch size
    """
    time = torch.tensor(viewpoint_camera.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1).detach()
    time = time*0. +viewpoint_camera.time
    
    means3D, rotation, opacity, colors, scales, invariance = process_overfit_Gaussians(pc, time)
    
    # Set up rasterization configuration
    #      return: colors, alphas, (normals, surf_normals, distort, median_depth, meta)
    rgb, alpha, meta = rendering_pass(means3D, rotation, scales, opacity, colors, invariance, viewpoint_camera, pc.active_sh_degree)
    rgb = rgb.squeeze(0).permute(2,0,1)
    
    # Masked loss
    mask = viewpoint_camera.sceneoccluded_mask.cuda()
    alpha = alpha.squeeze(-1)
    alpha_err = (mask - alpha).abs().mean()
    
    # Depth norm - Surface norm loss
    normal_err = (1 - (meta[0] * meta[1]).sum(dim=0))[None].mean()
    return rgb, meta[-1], normal_err, alpha_err



def render_coarse(viewpoint_camera, pc):
    """During the coarse stage we train the canonical gaussians on an un-lit scene
    """
    # Get gaussian parameters
    means3D, rotation, opacity, colors, scales, invariance = process_Gaussians(pc)
    
    # Set up rasterization configuration
    #      return: colors, alphas, (normals, surf_normals, distort, median_depth, meta)
    rgb, alpha, meta = rendering_pass(means3D, rotation, scales, opacity, colors, invariance, viewpoint_camera, pc.active_sh_degree)
    rgb = rgb.squeeze(0).permute(2,0,1)
        
    # Depth norm - Surface norm loss
    normal_err = (1 - (meta[0] * meta[1]).sum(dim=0))[None].mean()
    return rgb, meta[-1], normal_err

def render_extended(viewpoint_camera, pc, abc, texture, optix_runner):
    """Main Rendering function
    
    Algorithm:
        Rendering Equation is:  l.c + (1-l).(a.cr + (1-a).(cq)) (simplified from the one I have in my notebooks so that we take small steps)
        where...
            (1) l is the intensity of the base colo (we can explicitly regularize this based on frame-frame rgb differences, whereby small l should map small/no difference in color
            after relighting)
            (2) c is the base color of the Gaussian i.e. can be G-Rendered
            (3) a is the intensity of reflection
            (4) cr and cq are the colors sampled from the reflection and transmission function, i.e. R-Rendered
        Note that:
            - l should not be view dependant, I think
            - c can be rendered with the normal Gaussian Splatting function
            - a is view dependant so should be rendered with G-Render, but with a single color channel (where we input alpha)
            - l, a, c, should all be rendered with opacity = 1, this essentially forces l to model the intensity of the diffuse color (for now)
            later I intend to turn cq into (opacity).c + (1-opacity).cq, such that "internal reflection" is accounted for, so that l.c is does not explicitly
            represent the base color but the independance of the base color in the relighting set up - for now lets take small steps
            
            
    """
    # time = torch.tensor(viewpoint_camera.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1).detach()
    # time = time*0. +viewpoint_camera.time
    
    means3D, rotation, opacity, colors, scales, invariance, reflection, refraction = process_Gaussians_extended(pc)
    
    # 1. G-Render c and l seperately
    #   c
    base_color, _, meta = rendering_pass(means3D, rotation, scales, opacity, colors, invariance, viewpoint_camera, pc.active_sh_degree)
    #   l
    base_intensity, _ = rendering_base_intensity(means3D, rotation, scales, opacity, invariance, viewpoint_camera, pc.active_sh_degree)
    
    #   a
    reflection_intensity, _ = rendering_base_intensity(means3D, rotation, scales, opacity, reflection, viewpoint_camera, pc.active_sh_degree)
    
    refraction_intensity, _ = rendering_base_intensity(means3D, rotation, scales, opacity, refraction, viewpoint_camera, pc.active_sh_degree)

    # We now need to process the reflection and transmission (mainly different in how d_i is calculated)
    # 1. Reflection di' = di - 2 (di - n)n
    xi, d, norms = render_multi_feature(means3D, rotation, scales, opacity, viewpoint_camera, pc.active_sh_degree)
    di = d - 2.*(d-norms)*norms
    
    # 1. Refraction dr' = eta*di + (eta cosi - cosr)n, where eta = 1/n_material and cosi = -n.di and cosr = sqrt(1-eta^2 (1-cos^2i))
    n_mat = 1
    eta = 1/n_mat
    cosi = -(norms*d).sum(dim=-1, keepdim=True)
    dr = eta * d + (eta * cosi - torch.sqrt(torch.clamp( (1 - eta**2 * (1 - cosi**2)), min=1e-8))) * norms
    dr = dr / (dr.norm(dim=-1, keepdim=True) + 1e-8)

    # R-Render Pass
    reflection_color = r_render(xi, di, norms, abc, texture, optix_runner, means3D, pc, colors,opacity, update_verts=True)
    
    transmission_color = r_render(xi, dr, norms, abc, texture, optix_runner, means3D, pc, colors, opacity)
    
    # transmission_color =  r_render(xi, di, norms, abc, texture, optix_runner, means3D, rotation, opacity, colors, scales)
    
    # Rendering function
    rgb = (base_color*base_intensity) + (1.-base_intensity)*((reflection_intensity*reflection_color) + (1.-reflection_intensity)*(base_color*refraction_intensity + (1.-refraction_intensity)*transmission_color))
    # Set up rasterization configuration
    rgb = rgb.squeeze(0).permute(2,0,1)
    
    return rgb, meta[-1]