
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms as T

TRANSFORM = T.ToTensor()

import cv2

class Camera(nn.Module):
    def __init__(self,           
                 R, T, 
                 fx,fy,cx,cy,k1,k2,p1,p2,
                 uid,
                 
                 data_device = "cuda", 
                 time = 0,

                 width=None, height=None,

                 image_path=None,
                 canon_path=None,
                 sceneoccluded_path=None,
                 diff_path=None,
                
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 ):
        super(Camera, self).__init__()
        self.device = torch.device(data_device)


        self.uid = uid
        self.R = R * np.array([[1, -1, -1]])

        self.T = T
        self.time = time
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        K = np.array([ [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]  ])
        dist = np.array([k1, k2, p1, p2])
        self.K , _ = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 1)
        
        self.dist = dist
        self.K0 = K

        self.image_height = height
        self.image_width = width
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
  
        self.image_path=image_path
        self.canon_path=canon_path
        self.sceneoccluded_path=sceneoccluded_path
        self.diff_path = diff_path
        
        self.image = None
        self.canon = None
        self.sceneoccluded_mask = None
        self.diff_image = None
    
    
    def update_K(self):
        K = np.array([ [self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0,  0,  1]  ])
        self.K , _ = cv2.getOptimalNewCameraMatrix(K, self.dist, (self.image_width, self.image_height), 1)
        self.K0 = K
        
    @property
    def intrinsics(self): # Get the intrinsics matric
        return torch.from_numpy(self.K).float()

    @property
    def camera_center(self):
        return torch.from_numpy(self.T).float()
        
    @property
    def pose(self):# Get the c2w 
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return torch.from_numpy(Rt).cuda().float()

    @property
    def w2c(self):
        c2w = self.pose.unsqueeze(0)
        R = c2w[:, :3, :3]  # 3 x 3
        T = c2w[:, :3, 3:4]  # 3 x 1
        
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.transpose(1, 2)
        T_inv = -torch.bmm(R_inv, T)
        viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=torch.float)
        viewmat[:, 3, 3] = 1.0  # homogenous
        viewmat[:, :3, :3] = R_inv
        viewmat[:, :3, 3:4] = T_inv
        return viewmat[0]

    
    def load_image_from_flags(self, tag):
        if tag == "image":
            img = Image.open(self.image_path).convert("RGB")
            img = img.resize(
                (self.image_width, self.image_height),
                resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
            )            
            self.image = TRANSFORM(img)
        elif tag == "canon":
            img = Image.open(self.canon_path).convert("RGB")
            img = img.resize(
                (self.image_width, self.image_height),
                resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
            )            
            self.canon = TRANSFORM(img)
        elif tag == "scene_occluded":
            mask = Image.open(self.sceneoccluded_path).split()[-1]
            mask = mask.resize(
                (self.image_width, self.image_height),
                resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
            )

            self.sceneoccluded_mask = 1. - TRANSFORM(mask)
            
        elif tag == "differences":
            
            diff_image = torch.load(self.diff_path, map_location="cpu").unsqueeze(0)

            # resize (height=512, width=512)
            resized = F.interpolate(diff_image, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)

            # remove batch dimension again
            self.diff_image = resized.squeeze(0)
    
