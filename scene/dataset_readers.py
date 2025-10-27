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
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
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
    b_path:str
    diff_path:str

    uid: int    
    width: int
    height: int
    time : float
    feature: torch.Tensor
   
class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras:list
    video_cameras: list
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


import sys
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    image_names = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
        cam_info = CameraInfo(
            uid=uid, 
            R=R, T=T,
            
            fx=focal_length_x,
            fy=focal_length_y,
            cx=width/2.0, cy=height/2.0,
            width=width, height=height,

            image_path=image_path, 
            canon_path=None,
            so_path=None,
            b_path=None,
            diff_path = None,
            time = -1.,
            feature=None, 
        ) # default by monocular settings.
        
        
        cam_infos.append(cam_info)
        image_names.append(image_name)
    sys.stdout.write('\n')
    return cam_infos, image_names

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        n_points = positions.shape[0]
        normals = np.random.randn(n_points, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # normalize each to unit length

    return BasicPointCloud(points=positions, colors=colors, normals=normals)
    


def process_relighting_cams(path, unsorted_cams, N=100):
    path2canons = os.path.join(path, 'meta', 'canonical_0')
    path2masks = os.path.join(path, 'meta', 'masks')
    path2background = os.path.join(path, 'meta', 'backgrounds')
    path2images = os.path.join(path, 'images')

    relighting_cams = []
    background_paths = []
    cnt=0
    for idx, cam in enumerate(unsorted_cams):
        cam_fp_name = f"cam{(idx+1):02}"
        canon_path = os.path.join(path2canons, f"{cam_fp_name}.jpg")
        mask_path = os.path.join(path2masks, f"{cam_fp_name}.png")
        
        canon_path = canon_path
        so_path = mask_path
        
        cam_images_path = os.path.join(path2images, f'{cam_fp_name}')
        
        cam_image_names = sorted(os.listdir(cam_images_path))
        
        for jdx, im_name in enumerate(cam_image_names):
            image_path = os.path.join(cam_images_path, im_name)
            b_path = os.path.join(path2background, f"{(jdx+1):03}.jpg")
            
            relighting_cams.append(CameraInfo(
            uid=cnt, 
            R=cam.R, T=cam.T,
            
            fx=cam.fx,
            fy=cam.fy,
            cx=cam.cx, cy=cam.cy,
            width=cam.width, height=cam.height,

            image_path=image_path, 
            canon_path=canon_path,
            so_path=so_path,
            b_path=b_path,
            diff_path = None,
            
            time = float(jdx/N),
            feature=None, 
            ))
            
            cnt += 1
            
            if idx == 0:
                background_paths.append(b_path)
            
    return relighting_cams, background_paths
            
import open3d as o3d
def downsample_pointcloud_voxel_target(pcd, target_points=100_000, max_iter=10, verbose=True):
    """
    """
    # Convert to Open3D point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd.points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd.colors)
    pcd_o3d.normals = o3d.utility.Vector3dVector(pcd.normals)

    N = np.asarray(pcd.points).shape[0]

    # Initial voxel search range
    bbox = pcd_o3d.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
    voxel_min = diag / 5000.0  # start small
    voxel_max = diag / 50.0    # start large

    best_pcd = pcd_o3d
    best_diff = float("inf")

    for i in range(max_iter):
        voxel_size = (voxel_min + voxel_max) / 2.0
        pcd_ds = pcd_o3d.voxel_down_sample(voxel_size)
        n_points = len(pcd_ds.points)

        diff = abs(n_points - target_points)
        if diff < best_diff:
            best_pcd, best_diff = pcd_ds, diff

        if verbose:
            print(f"[{i+1}/{max_iter}] voxel={voxel_size:.5f} → {n_points} pts (target {target_points})")

        # Adjust search bounds
        if n_points > target_points:
            voxel_min = voxel_size
        else:
            voxel_max = voxel_size

        # Early stop if we're close enough
        if diff / target_points < 0.05:
            break

    # Convert back to BasicPointCloud
    return BasicPointCloud(
        points=np.asarray(best_pcd.points),
        colors=np.asarray(best_pcd.colors),
        normals=np.asarray(best_pcd.normals)
    )

def readColmapInfo(path, N=98, downsample=2):
    """Construct data frames for each typice
    
    Notes:
        - We need seperate references for the canonical and the re-lighting scenes
        1. We first need to load the canonical images, we take the last 19 images/cam_params, as these correspond to the re-lit cameras in our dataset
        2. Copy the info and extend to 
    
    """
    print("Reading colmap data ...")
    path2colmap = os.path.join(path, 'meta', 'colmap_0')
    
    try:
        cameras_extrinsic_file = os.path.join(path2colmap, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path2colmap, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path2colmap, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path2colmap, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    
    path2images = os.path.join(path2colmap, 'dense', 'images')
    cam_infos_unsorted, image_names = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path2images)
    sorted_pairs = sorted(zip(image_names, cam_infos_unsorted), key=lambda x: x[0])
    image_names, canon_cam_infos = zip(*sorted_pairs)
    
    nerf_normalization = getNerfppNorm(canon_cam_infos)
    
    relighting_cams, background_paths = process_relighting_cams(path, canon_cam_infos[-19:])
    
    test_idx_set = [i for i in range(10)]
    test_cams = [cam for idx, cam in enumerate(relighting_cams) if (idx % 100) in test_idx_set  ]
    relighting_cams = [cam for idx, cam in enumerate(relighting_cams) if (idx % 100) not in test_idx_set]
    
    M = N - len(test_idx_set)
    mip_splat_cams = [cam for i, cam in enumerate(relighting_cams) if i%M == 0]
    
    # Get sparse colmap point cloud
    ply_path = os.path.join(path2colmap, "dense/fused.ply")

    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    if pcd.points.shape[0] > 120000:
        pcd = downsample_pointcloud_voxel_target(pcd, target_points=50_000)
    
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=relighting_cams,
        test_cameras=test_cams,
        video_cameras=canon_cam_infos,
        mip_splat_cams=mip_splat_cams,
        
        ba_cameras=canon_cam_infos,
        
        nerf_normalization=nerf_normalization,
        background_pth_ids=background_paths,
        splats=None
    )
    return scene_info

def opengl_to_opencv(c2w: np.ndarray) -> np.ndarray:
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(c2w.dtype)
    return c2w @ flip

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
            b_path=None,
            diff_path=None,
            time=float(frame.get("time", -1.0)),
            feature=None,
        ))

    return cam_infos


def readCamerasFromCanon(path, canon_cams, M=19):
    
    background_path = os.path.join(path, 'meta', 'backgrounds')
    background_im_paths = [os.path.join(background_path, f) for f in sorted(os.listdir(background_path))]
    relit_path = os.path.join(path, 'meta', 'images')
    masks_path = os.path.join(path, 'meta', 'masks')

    # Get the colmap id for the first relit camera (i.e. the last 19 frames of the nerfstudio dataset)
    N = len(canon_cams) - M
    L = len(background_im_paths)

    relit_cams = []
    for cam in canon_cams:
        if cam.uid > N:
            cam_id = cam.uid - N
            cam_name = f'cam{cam_id:02}'
            
            for background_id, b_path in enumerate(background_im_paths):
                im_name = b_path.split('/')[-1].replace('png', 'jpg') # e.g. '000.jpg'
                im_path = os.path.join(relit_path, cam_name, im_name)
                mask_path = os.path.join(masks_path, f'{cam_name}.png')
                
                time = background_id / L
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
                    b_path=b_path,
                    diff_path = None,
                    time = time,
                    feature=None, 
                )
                relit_cams.append(cam_info)

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

                    image_path=None, 
                    canon_path=None,
                    so_path=None,
                    b_path=None,
                    diff_path = None,
                    time=float(idx / 99),
                    feature=None, 
                )
        
        cams.append(cam_info)
    return cams

def readNerfstudioInfo(path, N=98, downsample=2):
    """Construct dataset from nerfstudio
    """
    print("Reading nerfstudio data ...")
    # Read camera transforms    
    canon_cam_infos = readCamerasFromTransforms(path, 'transforms.json')
    
    cam_infos, background_paths, L = readCamerasFromCanon(path, canon_cam_infos) # L is the number of background paths
    
    # split into training and test dataset
    V_cam = 14
    L_test_idx_set = [8, 12, 27, 36, 48, 55, 61, 71, 81, 98] # The lighting-only test set
    V_test_idx_set = [(V_cam*100)+i for i in range(100) if i not in L_test_idx_set] # The novel-view only test set
    LV_test_idx_set = [(V_cam*100)+i for i in range(100) if i in L_test_idx_set] # The novel-view & novel lighting test set
    
    L_test_cams = [cam for idx, cam in enumerate(cam_infos)  if (idx % L) in L_test_idx_set and idx not in LV_test_idx_set] # For indexs n the lighting test set
    V_test_cams = [cam for idx, cam in enumerate(cam_infos) if idx in V_test_idx_set] # For indexs in the novel view test set
    LV_test_cams = [cam for idx, cam in enumerate(cam_infos) if idx in LV_test_idx_set] # For indexs in the novel view and novel lighting test set
    test_cams = [L_test_cams, V_test_cams, LV_test_cams]
    
    relighting_cams = [cam for idx, cam in enumerate(cam_infos) if (idx % L) not in L_test_idx_set and idx not in V_test_idx_set] # For indexs not in lighting and novel view cameras
    
    video_cams = generate_circular_cams(path, cam_infos[V_cam*100])
    
    
    nerf_normalization = getNerfppNorm(relighting_cams)
    
    
    scene_info = SceneInfo(
        train_cameras=relighting_cams,
        test_cameras=test_cams,
        video_cameras=video_cams,
        
        nerf_normalization=nerf_normalization,
        background_pth_ids=background_paths,

        param_path=os.path.join(path, 'splat','splat.ply'),

    )
    return scene_info

sceneLoadTypeCallbacks = {
    "colmap": readColmapInfo,
    "nerfstudio": readNerfstudioInfo
}
