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
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras:list
    ba_cameras:list
    video_cameras: list
    nerf_normalization: dict
    background_pth_ids:list

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
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])

    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            
    return cam_infos

from scipy.spatial.transform import Rotation
OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def readStudioCams(path, cams2world, focal, H, W, xyz, N, downsample=2):    
    path2imgs = os.path.join(path, 'images')
    path2diff = os.path.join(path, 'differences')
    cams = sorted(os.listdir(path2imgs))

    # upsample intrinsics to match the original image
    focal = focal * 3.75
    W = 1920
    H = 1080
    
    fovx = focal2fov(focal, W)
    fovy = focal2fov(focal, H)
    W = 1920 // downsample
    H = 1080 // downsample
    cam_infos = []

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    global_T = np.linalg.inv(cams2world[0] @ OPENGL @ rot)
    cams2world_aligned = [ (global_T @ c2w) for c2w in cams2world ]
    
    pts_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    global_T = np.linalg.inv(cams2world[0] @ OPENGL @ rot)

    xyz = (global_T @ pts_h.T).T[:, :3]
    im_name = '000.png'
    background_pth_ids = []
    cnt = 0
    for idx, pose in enumerate(cams2world_aligned):
        for ids, im_name in enumerate(sorted(os.listdir(os.path.join(path2imgs, cams[idx])))):
            image_path = os.path.join(path2imgs, cams[idx], im_name)
            so_path = os.path.join(path, "meta", "masks", f"{cams[idx]}.png")
            canon_path = os.path.join(path, "meta", "canonical_0", f"{cams[idx]}.jpg")
            b_path = os.path.join(path, "meta", "backgrounds", f"{im_name[:-4]}.png")
            diff_path = os.path.join(path2diff, cams[idx], im_name.replace('.jpg', '.pt'))
            
            w2c = np.linalg.inv(pose)
            R = pose[:3,:3] # rotation
            T = w2c[:3,3] 
            
            cx =  W / 2.0
            cy = H / 2.0
            fx = 0.5 * W / np.tan(fovx * 0.5)
            fy = 0.5 * H / np.tan(fovy * 0.5)
            
            w = W
            h = H

            cam_infos.append(
                CameraInfo(
                    R=R, T=T,
                    cx=cx, cy=cy, fx=fx, fy=fy,
                    width=w, height=h,
                    
                    image_path=image_path, 
                    so_path=so_path,
                    b_path=b_path,
                    diff_path=diff_path,
                    canon_path=canon_path,
                    
                    uid=cnt,
                    time = float(ids%N)/N, feature=0.
                )
            )
            background_pth_ids.append(b_path)
            cnt +=1
    return cam_infos, xyz, background_pth_ids


def readCanonicalCams(path, cams2world, focal, H, W, downsample=2):    
    path2imgs = os.path.join(path, 'meta', 'canonical_0')
    cams = sorted(os.listdir(path2imgs))

    # upsample intrinsics to match the original image
    focal = focal * 3.75
    W = 1920
    H = 1080
    fovx = focal2fov(focal, W)
    fovy = focal2fov(focal, H)
    W = 1920 //downsample
    H = 1080 //downsample
    
    cam_infos = []

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    global_T = np.linalg.inv(cams2world[0] @ OPENGL @ rot)
    cams2world_aligned = [ (global_T @ c2w) for c2w in cams2world ]
    
    for idx, pose in enumerate(cams2world_aligned):
        # Construct path to canonical image
        image_path = os.path.join(path2imgs, f"{cams[idx]}")
        so_path = os.path.join(path, "meta", "masks", f"{cams[idx].replace('.jpg', '.png')}")

        # Define the camera calibration settings
        w2c = np.linalg.inv(pose)
        R = pose[:3,:3]
        T = w2c[:3,3] 
        
        cx =  W / 2.0
        cy = H / 2.0
        fx = 0.5 * W / np.tan(fovx * 0.5)
        fy = 0.5 * H / np.tan(fovy * 0.5)
        
        w = W
        h = H
        cam_infos.append(
            CameraInfo(
                R=R, T=T,
                cx=cx, cy=cy, fx=fx, fy=fy,
                width=w, height=h,
                
                image_path=image_path, 
                so_path=so_path,
                b_path=None,
                diff_path=None,
                
                uid=idx,
                time = 0., feature=0.
            )
        )

    return cam_infos


def readHomeStudioInfo(path, N=98, downsample=2):
    print("Reading Training Data")
    # Load camera data (generated from mast3r)
    meta_pth = os.path.join(path, 'meta', 'info.json')
    with open(meta_pth, "r") as f:
        meta = json.load(f)

    # Camera pose data
    focal = meta['focals'][0] # shared intrinsics
    W,H = meta['imsize'][0][0], meta['imsize'][0][1]
    poses = meta['cams2world']
    c2w = []
    for pose in poses:
        c2w.append(np.array(pose))
    
    # Load initial point cloud and colors
    pcd_path = os.path.join(path, 'meta', 'pointcloud.npy')
    pcd = np.load(pcd_path, allow_pickle=True)
    pts = pcd[::5, :3]
    col = pcd[::5, 3:]

    # Get training cameras
    train_cams, pts, background_pth_ids = readStudioCams(path, c2w, focal, H, W, pts, N, downsample)
    
    # TODO: Define testing method
    # For now just use the first camera as test
    test_cams = [cam for i, cam in enumerate(train_cams) if i%N in [0, 1, 2]]
    train_cams = [cam for i, cam in enumerate(train_cams) if i%N not in [0, 1, 2]]
    
    # We also need to define a dataset for bundle adjusting the screen and cams
    BA_cams = [train_cams[i] for i in range(len(train_cams)) if i % N == 0]

    nerf_normalization = getNerfppNorm(train_cams)
    
    pcd = BasicPointCloud(points=pts, colors=col, normals=np.zeros((pts.shape[0], 3)))

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        video_cameras=train_cams,

        ba_cameras=BA_cams,
        
        nerf_normalization=nerf_normalization,
        background_pth_ids=background_pth_ids
    )
    return scene_info

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

def readStudio4Info(path, N=98, downsample=2):
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
    
    
    path2images = os.path.join(path, 'meta', 'canonical_1_pose_estimation')
    cam_infos_unsorted, image_names = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path2images)
    sorted_pairs = sorted(zip(image_names, cam_infos_unsorted), key=lambda x: x[0])
    image_names, canon_cam_infos = zip(*sorted_pairs)
    
    nerf_normalization = getNerfppNorm(canon_cam_infos)
    
    relighting_cams, background_paths = process_relighting_cams(path, canon_cam_infos[-19:])
    
    test_idx_set = [i for i in range(10)]
    test_cams = [cam for idx, cam in enumerate(relighting_cams) if (idx % 100) in test_idx_set  ]
    relighting_cams = [cam for idx, cam in enumerate(relighting_cams) if (idx % 100) not in test_idx_set]
    
    # Get sparse colmap point cloud
    ply_path = os.path.join(path2colmap, "dense/fused.ply")

    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    if pcd.points.shape[0] > 120000:
        pcd = downsample_pointcloud_voxel_target(pcd, target_points=200_000)
    
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=relighting_cams,
        test_cameras=test_cams,
        video_cameras=canon_cam_infos,

        ba_cameras=canon_cam_infos,
        
        nerf_normalization=nerf_normalization,
        background_pth_ids=background_paths
    )
    return scene_info

sceneLoadTypeCallbacks = {
    "homestudio": readStudio4Info
}
