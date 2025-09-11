#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from utils.graphics_utils import getWorld2View2
from PIL import Image
from torchvision import transforms as T

TRANSFORM = T.ToTensor()


# def getProjectionMatrix_(znear, zfar, fovX, fovY, ppx, ppy, image_width, image_height):
#     """
#     Generate a perspective projection matrix incorporating the principal point (ppx, ppy).

#     Parameters:
#         znear (float): Distance to the near clipping plane.
#         zfar (float): Distance to the far clipping plane.
#         fovX (float): Horizontal field of view in radians.
#         fovY (float): Vertical field of view in radians.
#         ppx (float): Principal point x-coordinate (image center x).
#         ppy (float): Principal point y-coordinate (image center y).
#         image_width (float): Width of the image.
#         image_height (float): Height of the image.

#     Returns:
#         torch.Tensor: A 4x4 perspective projection matrix.
#     """
#     tanHalfFovY = math.tan(fovY / 2)
#     tanHalfFovX = math.tan(fovX / 2)

#     top = tanHalfFovY * znear
#     bottom = -top
#     right = tanHalfFovX * znear
#     left = -right

#     # Adjust the projection based on the principal point
#     offset_x = 2 * (ppx / image_width) - 1  # Normalized offset in [-1, 1]
#     offset_y = 2 * (ppy / image_height) - 1  # Normalized offset in [-1, 1]

#     # Initialize the projection matrix
#     P = torch.zeros(4, 4)

#     z_sign = 1.0  # Use 1.0 for a right-handed coordinate system

#     # Scale factors for x and y
#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)

#     # Translation offsets
#     P[0, 2] = (right + left) / (right - left) - offset_x
#     P[1, 2] = (top + bottom) / (top - bottom) - offset_y

#     # Depth scaling and perspective division
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)

#     return P

def getProjectionMatrixFromIntrinsics(fx, fy, cx, cy, width, height, znear, zfar):
    P = torch.zeros(4, 4)

    # Map intrinsics to normalized device coordinates
    P[0, 0] = 2 * fx / width
    P[0, 2] = 1 - 2 * cx / width
    P[1, 1] = 2 * fy / height
    P[1, 2] = 2 * cy / height - 1

    # Depth (Z)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = (-znear * zfar) / (zfar - znear)
    P[3, 2] = 1.0

    return P

class Camera(nn.Module):
    def __init__(self,           
                 R, T, 
                 fx,fy,cx,cy, 
                 uid,
                 
                 data_device = "cuda", 
                 time = 0,

                 width=None, height=None,

                 image_path=None,
                 sceneoccluded_path=None,
                 scene_path=None,
                 glass_path=None,
                 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 ):
        super(Camera, self).__init__()
        self.device = torch.device(data_device)


        self.uid = uid
        self.R = R
        self.T = T
        self.time = time
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
                
        self.image_height = height
        self.image_width = width
        
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        
        self.projection_matrix = getProjectionMatrixFromIntrinsics(
            self.fx, self.fy, self.cx, self.cy, self.image_width, self.image_height,
            self.znear, self.zfar
        ).transpose(0, 1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        #
        self.image_path=image_path
        self.sceneoccluded_path=sceneoccluded_path
        self.scene_path=scene_path
        self.glass_path=glass_path
        
        self.image = None
        self.sceneoccluded_mask = None
        self.scene_mask = None
        self.glass_mask = None
        
    @property
    def intrinsics(self): # Get the intrinsics matric
        return torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0,  0,      1]
        ], dtype=torch.float32)
        
    @property
    def pose(self):# Get the c2w 
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0

        C2W = torch.from_numpy(np.linalg.inv(Rt)).cuda().float()
        return C2W

    def generate_rays(self, c2w=None, H=None, W=None, fx=None, fy=None, cx=None, cy=None, device="cuda"):
        if H is None or W is None:
            H, W = self.image_height, self.image_width

        if c2w is None:
            c2w = self.world_view_transform.transpose(0, 1).inverse().to(device)

        # Pixel grid
        i = torch.arange(W, device=device, dtype=torch.float32)
        j = torch.arange(H, device=device, dtype=torch.float32)
        jj, ii = torch.meshgrid(j, i, indexing="ij")  # [H,W]

        # Camera-space ray directions (OpenGL convention: +y up, -z forward)
        x = -(ii - cx) / fx
        y = -(jj - cy) / fy
        z = -torch.ones_like(x)
        dirs_cam = torch.stack([x, y, z], dim=-1)  # [H,W,3]
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

        # Transform to world space
        dirs_world = dirs_cam @ c2w[:3, :3].T
        dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
        origins_world = c2w[:3, 3].expand_as(dirs_world)

        return origins_world, dirs_world

    def surface_sample(self, origin, direction, abc, texture):
        # origins_world, dirs_world
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

        uv = torch.stack([u_coord, v_coord], dim=-1)  # (N,2)
        uv = (2.0 * uv - 1.0).unsqueeze(0).unsqueeze(2)

        sampled = F.grid_sample(
            texture.unsqueeze(0), uv,
            align_corners=True, mode="bilinear"
        ).squeeze(0).squeeze(-1).reshape(3, H, W)

        return sampled
    
    def update_projections(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        
        self.projection_matrix = getProjectionMatrixFromIntrinsics(
            self.fx, self.fy, self.cx, self.cy, self.image_width, self.image_height,
            self.znear, self.zfar
        ).transpose(0, 1)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def load_image_from_flags(self, tag):
        if tag == "image":
            self.image = TRANSFORM(
                Image.open(self.image_path).convert("RGB")
            )
        elif tag == "scene":
            self.scene_mask = TRANSFORM(
                Image.open(self.scene_path).split()[-1]
            )
        elif tag == "scene_occluded":
            self.sceneoccluded_mask = TRANSFORM(
                Image.open(self.sceneoccluded_path).split()[-1]
            )
        elif tag == "scene":
            self.glass_mask = TRANSFORM(
                Image.open(self.glass_path).split()[-1]
            )
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

