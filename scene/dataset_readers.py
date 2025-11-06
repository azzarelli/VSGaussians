import os
from PIL import Image

from typing import NamedTuple

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
# from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    fx: np.array
    fy: np.array
    cx: np.array
    cy: np.array
    
    k1: np.array
    k2: np.array
    p1: np.array
    p2: np.array

    image_path: str
    canon_path:str
    so_path: str
    
    image: torch.Tensor
    canon:torch.Tensor
    mask: torch.Tensor


    uid: int    
    width: int
    height: int
    time : int
   
class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras:list
    video_cameras: list
    ba_background_fp:str
    nerf_normalization: dict
    background_pth_ids:list
    param_path:str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal #* 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}


def readCamerasFromTransforms(path, transformsfile):
    cam_infos = []

    tf_path = os.path.join(path, transformsfile)
    with open(tf_path, "r") as json_file:
        contents = json.load(json_file)

    # Global intrinsics
    g_fx = contents.get("fl_x")
    g_fy = contents.get("fl_y")
    g_cx = contents.get("cx")
    g_cy = contents.get("cy")
    g_w  = contents.get("w")
    g_h  = contents.get("h")
    g_k1 = contents.get("k1")
    g_k2 = contents.get("k2")
    g_p1 = contents.get("p1")
    g_p2 = contents.get("p2")

    # Nerfstudio normalization (transform + scale)
    frames = contents["frames"] 
    for idx, frame in enumerate(frames):
        fx = frame.get("fl_x", g_fx)
        fy = frame.get("fl_y", g_fy)
        cx = frame.get("cx", g_cx)
        cy = frame.get("cy", g_cy)
        w  = frame.get("w", g_w)
        h  = frame.get("h", g_h)

        k1 = frame.get("k1", g_k1)
        k2 = frame.get("k2", g_k2)
        p1 = frame.get("p1", g_p1)
        p2 = frame.get("p2", g_p2)

        # Load and convert transform
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
   
        
        R = c2w[:3, :3]
        T = c2w[:3, 3]

        image_path = os.path.normpath(os.path.join(path, frame["file_path"]))

        cam_infos.append(CameraInfo(
            uid=frame.get("colmap_im_id", idx),
            R=R, T=T,
            fx=fx, fy=fy, cx=cx, cy=cy,
            k1=k1, k2=k2, p1=p1, p2=p2,
            width=w, height=h,
            image_path=image_path,
            canon_path=None,
            so_path=None,
            image=None,
            canon=None,
            mask=None,
            time=float(frame.get("time", -1.0)),
        ))

    return cam_infos

from torchvision import transforms as T
TRANSFORM = T.ToTensor()

def readCamerasFromCanon(path, canon_cams, M=19, preload_gpu=False):
    
    background_path = os.path.join(path, 'meta', 'backgrounds')
    background_im_paths = [os.path.join(background_path, f) for f in sorted(os.listdir(background_path))]
    relit_path = os.path.join(path, 'meta', 'images')
    masks_path = os.path.join(path, 'meta', 'masks')

    # Get the colmap id for the first relit camera (i.e. the last 19 frames of the nerfstudio dataset)
    N = len(canon_cams) - M
    L = len(background_im_paths)
    
    image=None
    canon=None
    mask=None

    relit_cams = []
    for cam in canon_cams:
        if cam.uid > N:
            cam_id = cam.uid - N
            cam_name = f'cam{cam_id:02}'
            
            for background_id, b_path in enumerate(background_im_paths):
                im_name = b_path.split('/')[-1].replace('png', 'jpg') # e.g. '000.jpg'
                im_path = os.path.join(relit_path, cam_name, im_name)
                mask_path = os.path.join(masks_path, f'{cam_name}.png')
                
                # Load 
                if preload_gpu:
                    img = Image.open(im_path).convert("RGB")
                    img = img.resize(
                        (cam.width, cam.height),
                        resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
                    )            
                    image = TRANSFORM(img).cuda()
                    
                    img = Image.open(cam.image_path).convert("RGB")
                    img = img.resize(
                        (cam.width, cam.height),
                        resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
                    )            
                    canon = TRANSFORM(img).cuda()
                    img = Image.open(mask_path).split()[-1]
                    img = img.resize(
                        (cam.width, cam.height),
                        resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
                    )

                    mask = 1. - TRANSFORM(img).cuda()

                time = background_id

                cam_info = CameraInfo(
                    uid=cam.uid, 
                    R=cam.R, T=cam.T,
                    
                    fx=cam.fx,
                    fy=cam.fy,
                    cx=cam.cx, cy=cam.cy,
                    k1=cam.k1, k2=cam.k2, p1=cam.p1, p2=cam.p2,

                    width=cam.width, height=cam.height,

                    image_path=im_path, 
                    canon_path=cam.image_path,
                    so_path=mask_path,
                    
                    image=image,
                    canon=canon,
                    mask=mask,
                    
                    time = time,
                )
                relit_cams.append(cam_info)
            break
    return relit_cams, background_im_paths, L

import math
def generate_circular_cams(
    path,
    cam,
):
    """
    Generate a circular camera path (all c2w) around a fixed point in front
    of the input camera, keeping orientation toward that point.
    Assumes OpenCV-style camera with forward = -Z in c2w.
    """

    # --- extract camera basis (columns of R are world-space camera axes)
    fp = os.path.join(path, 'meta/video_paths.json')
    with open(fp, "r") as json_file:
        contents = json.load(json_file)

    zoomscale = 0.65
    # --- recompute fx, fy from that new FOV
    fx_new = cam.fx*zoomscale
    fy_new = cam.fy*zoomscale
    cams=[]
    for idx, info in enumerate(contents['camera_path']):
        #
        c2w = np.array(info["camera_to_world"], dtype=np.float32).reshape(4, 4)

        R = c2w[:3, :3]
        T = c2w[:3, 3]

        cam_info = CameraInfo(
                    uid=cam.uid, 
                    R=R, T=T,
                    
                    fx=fx_new,
                    fy=fy_new,
                    cx=cam.cx, cy=cam.cy,
                    k1=cam.k1, k2=cam.k2, p1=cam.p1, p2=cam.p2,

                    width=cam.width, height=cam.height,
                    image=None,
                    canon=None,
                    mask=None,
                    image_path=None, 
                    canon_path=None,
                    so_path=None,
                    time=idx,
                )
        
        cams.append(cam_info)
    return cams

def readNerfstudioInfo(path, N=98, preload_imgs=False):
    """Construct dataset from nerfstudio
    """
    print("Reading nerfstudio data ...")
    # Read camera transforms    
    canon_cam_infos = readCamerasFromTransforms(path, 'transforms.json')
    
    cam_infos, background_paths, L = readCamerasFromCanon(path, canon_cam_infos, preload_gpu=preload_imgs) # L is the number of background paths
    
    # split into training and test dataset
    V_cam = 14
    L_test_idx_set = [8, 12, 27, 36, 48, 55, 61, 71, 81, 98] # The lighting-only test set
    V_test_idx_set = [(V_cam*100)+i for i in range(100) if i not in L_test_idx_set] # The novel-view only test set
    LV_test_idx_set = [(V_cam*100)+i for i in range(100) if i in L_test_idx_set] # The novel-view & novel lighting test set
    
    # Load the split datasets
    L_test_cams = [cam for idx, cam in enumerate(cam_infos)  if (idx % L) in L_test_idx_set and idx not in LV_test_idx_set] # For indexs n the lighting test set
    V_test_cams = [cam for idx, cam in enumerate(cam_infos) if idx in V_test_idx_set] # For indexs in the novel view test set
    LV_test_cams = [cam for idx, cam in enumerate(cam_infos) if idx in LV_test_idx_set] # For indexs in the novel view and novel lighting test set
    test_cams = [L_test_cams, V_test_cams, LV_test_cams]

    relighting_cams = [cam for idx, cam in enumerate(cam_infos) if (idx % L) not in L_test_idx_set and idx not in V_test_idx_set] # For indexs not in lighting and novel view cameras
    
    # Select cameras with a common background for pose estimation (from the training set)
    selected_background_fp = background_paths[0]

    # Camera path for novel view
    video_cams = None #generate_circular_cams(path, cam_infos[V_cam*100])
    nerf_normalization = getNerfppNorm(relighting_cams)
    
    
    scene_info = SceneInfo(
        train_cameras=relighting_cams,
        test_cameras=test_cams,
        video_cameras=video_cams,
        
        ba_background_fp=selected_background_fp,
        
        nerf_normalization=nerf_normalization,
        background_pth_ids=background_paths,

        param_path=os.path.join(path, 'splat','splat.ply'),

    )
    return scene_info

sceneLoadTypeCallbacks = {
    "nerfstudio": readNerfstudioInfo
}
