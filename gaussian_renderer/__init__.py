import torch
import math
import torch.nn.functional as F

from gsplat.rendering import rasterization



def render_IBL_source(cam, abc, texture, device="cuda"):
    intr = cam.intrinsics
    fx,fy,cx,cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    origin, direction = cam.generate_rays(None, None, None, fx,fy,cx,cy)
    
    return cam.surface_sample(origin, direction, abc, texture.cuda())


def process_Gaussians(pc, time):
    means3D = pc.get_xyz
    colors = pc.get_features
    opacity = pc.get_opacity
    try:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc._rotation
        

        # means3D, rotations, opacity, colors, extras = pc._deformation(
        #     point=means3D, 
        #     rotations=rotations,
        #     scales=scales,
        #     times_sel=feature, 
        #     h_emb=opacity,
        #     shs=colors,
        # )
        
        opacity = pc.get_fine_opacity_with_3D_filter(opacity[:,0].unsqueeze(-1))
        rotations = pc.rotation_activation(rotations)
        return means3D, rotations, opacity, colors, scales
    except:
        scales = torch.zeros_like(means3D, device=means3D.device) + 0.001
        opacity = torch.ones_like(means3D[:,0].unsqueeze(-1), device=means3D.device)
        
        rotations = pc.rotation_activation(pc._rotation)
        return means3D, rotations, opacity, colors, scales
 
def rendering_pass(means3D, rotation, scales, opacity, colors, cam, sh_deg=3, mode="RGB"):
    return rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        cam.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
        cam.intrinsics.unsqueeze(0).cuda(),
        cam.image_width, 
        cam.image_height,
        
        render_mode=mode,
        rasterize_mode='antialiased',
        eps2d=0.1,
        sh_degree=sh_deg #pc.active_sh_degree
    )

def render(viewpoint_camera, pc, abc, texture, view_args=None):
    """
    Render the scene for viewing
    """
    extras = None

    time = torch.tensor(viewpoint_camera.time).to(pc._xyz.device).repeat(pc._xyz.shape[0], 1)
    means3D, rotation, opacity, colors, scales = process_Gaussians(pc, time)
    
    # Set arguments depending on type of viewing
    if view_args['vis_mode'] in 'render':
        mode = "RGB"
    elif view_args['vis_mode'] == 'alpha':
        mode = "RGB"
    elif view_args['vis_mode'] == 'D':
        mode = "D"
    elif view_args['vis_mode'] == 'ED':
        mode = "ED"
    
    # Render
    render, alpha, _ = rendering_pass(
        means3D, rotation, scales, opacity, colors, viewpoint_camera, 
        pc.active_sh_degree,
        mode=mode
    )
    
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


from utils.loss_utils import l1_loss

def render_batch(viewpoint_cams, pc):
    """Training renderer the scene"""
    
    L1 = 0.

    time = torch.tensor(viewpoint_cams[0].time).to(pc._xyz.device).repeat(pc._xyz.shape[0], 1).detach()
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        time = time*0. +viewpoint_camera.time
        
        means3D, rotation, opacity, colors, scales = process_Gaussians(pc, time)
                        
        # Set up rasterization configuration
        rgb, _, _ = rendering_pass(means3D, rotation, scales, opacity, colors, viewpoint_camera, pc.active_sh_degree)
        rgb = rgb.squeeze(0).permute(2,0,1)
        gt_img = viewpoint_camera.image.cuda() *(viewpoint_camera.sceneoccluded_mask.cuda())
        
        
        L1 += l1_loss(rgb, gt_img)
           
    return L1

