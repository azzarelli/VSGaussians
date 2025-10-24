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
        self.R = R
        self.T = T
        self.time = time
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        K = np.array([ [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]  ])
        dist = np.array([k1, k2, p1, p2])
        self.K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 1)
        
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
        
    @property
    def intrinsics(self): # Get the intrinsics matric
        return torch.from_numpy(self.K).float()

        
    @property
    def pose(self):# Get the c2w 
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0

        C2W = torch.from_numpy(Rt).cuda().float()
        return C2W

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
            # later, in main process:
            # self.diff_image = self.diff_image.to("cuda")
            # self.diff_image = torch.load(self.diff_path, map_location="cuda")
            # print(self.diff_path)
            # with open(self.diff_path, "rb") as f:
            #     buffer = io.BytesIO(f.read())
            # self.diff_image = torch.load(buffer, map_location="cuda")