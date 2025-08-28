import torch
import math
import torch.nn.functional as F

from gsplat.rendering import rasterization

def render(viewpoint_camera, pc,
           stage="fine", view_args=None):
    """
    Render the scene.
    """

    extras = None

    means3D_ = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D_.device).repeat(means3D_.shape[0], 1)

    colors = pc.get_features

    opacity = pc.get_opacity.clone()

    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation


    means3D, rotations, opacity, colors, extras = pc._deformation(
        point=means3D_, 
        rotations=rotations,
        scales=scales,
        times_sel=time, 
        h_emb=opacity,
        shs=colors,
        view_dir=viewpoint_camera.direction_normal(),
    )
    
    opacity = pc.get_fine_opacity_with_3D_filter(opacity)
    rotation = pc.rotation_activation(rotations)

    view_args= {'vis_mode':'render'}

            
    # print(.shape, means3D.shape)
    rendered_image, rendered_depth, norms = None, None, None
    if view_args['vis_mode'] in ['render']:

        rendered_image, alpha, _ = rasterization(
                        means3D, rotation, scales, opacity.squeeze(-1), colors,

            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
        
    elif view_args['vis_mode'] == 'alpha':
        _, rendered_image, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='RGB',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'D':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='D',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
        
    elif view_args['vis_mode'] == 'ED':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='ED',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    
    
    # A = (458, 622)
    # B = (3819, 511)
    # C = (3999,2504)
    # D = (375, 2554)
    # verts = [A,B,C,D]
    # verts_ = []
    # for v in verts:
    #     verts_.append((v[0]/2, v[1]/2))
    # mask = quad_mask(rendered_image.shape[1], rendered_image.shape[2],verts_, device=rendered_image.device).unsqueeze(0)
    
    # warped = warp_into_quad(viewpoint_camera.mask, verts_, out_size=(rendered_image.shape[1], rendered_image.shape[2]))
    # rendered_image = rendered_image * (~mask) + warped * mask

    return {
        "render": rendered_image,
        "extras":extras # A dict containing mor point info
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }


import torchvision.transforms.functional as FF
from torchvision.transforms import InterpolationMode
def warp_into_quad(src, dst_vertices, out_size, interpolation=InterpolationMode.BILINEAR):
    """
    src: (C,H2,W2) tensor image
    dst_vertices: list of 4 (x,y) coords in output space (top-left, top-right, bottom-right, bottom-left)
    out_size: (H, W) size of the destination canvas
    """
    C, H2, W2 = src.shape
    # Source points
    src_pts = [(0,0), (W2-1,0), (W2-1,H2-1), (0,H2-1)]
    
    # Torchvision expects lists of lists
    src_pts = [list(map(float, pt)) for pt in src_pts]
    dst_pts = [list(map(float, pt)) for pt in dst_vertices]
    
    # Torchvision wants CHW in a PIL-like wrapper, so add batch
    src_bchw = src.unsqueeze(0).cuda()  # (1,C,H2,W2)
    
    # Apply perspective
    warped = FF.perspective(
        img=src_bchw,
        startpoints=src_pts,
        endpoints=dst_pts,
        interpolation=interpolation,
        fill=0
    )
    return warped.squeeze(0) 

def quad_mask(H, W, vertices, device="cuda"):
    """
    vertices: list of 4 (x,y) tuples in clockwise or counter-clockwise order
              e.g. [(xA,yA),(xB,yB),(xC,yC),(xD,yD)]
    """
    # Unpack
    vx = torch.tensor([v[0] for v in vertices], device=device, dtype=torch.float32)
    vy = torch.tensor([v[1] for v in vertices], device=device, dtype=torch.float32)

    # Make grid of pixel coords
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )

    # Ray casting algorithm: check intersections of horizontal ray
    inside = torch.zeros((H, W), device=device, dtype=torch.bool)
    j = -1
    for i in range(4):
        xi, yi = vx[i], vy[i]
        xj, yj = vx[j], vy[j]
        cond = ((yi > yy) != (yj > yy)) & \
               (xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-6) + xi)
        inside ^= cond
        j = i

    return inside  # shape (H,W), bool mask

from utils.loss_utils import l1_loss

def render_batch(viewpoint_cams, pc):
    """Training renderer the scene"""
    
    means3D = pc.get_xyz
    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation
    colors = pc.get_features
    opacity = pc.get_opacity
    
    L1 = 0.

    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        time = time*0. +viewpoint_camera.time
        
        
        means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
            point=means3D, 
            rotations=rotations,
            scales = scales,
            times_sel=time, 
            h_emb=opacity,
            shs=colors,
            view_dir=viewpoint_camera.direction_normal(),
        )
                
        opacity_final = pc.get_fine_opacity_with_3D_filter(opacity_final)        
        rotations_final = pc.rotation_activation(rotations_final)
        
        scales_final = scales
        
        # Set up rasterization configuration
        rgb, alpha, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1), colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        gt_img = viewpoint_camera.original_image.cuda()
        L1 += l1_loss(rgb, gt_img)
   
    return L1

