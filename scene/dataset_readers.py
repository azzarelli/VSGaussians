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
    print("hello!!!!")
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

def generateSingleViewCameras(path, verts, fx=1528,fy=1538, h=3000, w=4000, ):
    import clip
    from PIL import Image
    print('Loading CLIP encoder')
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    
    images = os.listdir(path) # Get image fo;es
    fovx = focal2fov(fx, w)
    fovy = focal2fov(fy, h)

    cam_infos = []
    F_max = float(len(images)-1 )
    for idx, frame in enumerate(images):
        time = float(idx)/F_max
        
        R = np.eye(3)
        T = np.zeros(3) # Z-direction +0.1 is backwards
        
        image_path = os.path.join(path, frame)
        image = Image.open(image_path.replace("train", "backgrounds"))

        # Load CLIP embedding
        CLIP_image = preprocess(image).unsqueeze(0).to("cuda")

        CLIP_image *= 0
        CLIP_image[:, 0] = 1.
        
        # Display CLIP processed image
        # import matplotlib.pyplot as plt
        # img = CLIP_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # # If values are in [0,1] it's fine, but if in [-1,1] or [0,255], normalize
        # if img.min() < 0:
        #     img = (img - img.min()) / (img.max() - img.min())
        # elif img.max() > 1.0:
        #     img = img / 255.0
        
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()
        # CLIP_image *= 0
        # CLIP_image[:, 0] = 1.
        
        with torch.no_grad():
            image_features = model.encode_image(CLIP_image).float().cpu() # (1, 512)
            
        # Load torch image on CPU
        # image = PILtoTorch(image,(w,h))
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=None,verts=verts,
                        image_path=image_path, image_name=frame, width=w, height=h,
                        time = time, mask=None, feature=image_features))
    
    del model, preprocess
    torch.cuda.empty_cache()
    
    return cam_infos, R, T, fovx, fovy

def circle_in_image_plane(R=np.eye(3), C0=np.zeros(3), cam_infos=None, r=1.0, n_steps=70):
    # Camera local axes in world space
    u = R @ np.array([1.0, 0.0, 0.0])  # right
    v = R @ np.array([0.0, 1.0, 0.0])  # up

    thetas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
    poses = []

    d = np.zeros(3)


    for idx, th in enumerate(thetas):
        cam = cam_infos[idx % len(cam_infos)]
        # Camera center in world coordinates on the circle
        C = C0 + r*np.cos(th)*u + r*np.sin(th)*v
        # Keep orientation fixed
        R_wc = R.copy()
        # OpenCV extrinsics: x_c = R x_w + t, with t = -R*C
        t = -R_wc @ C

        # 4x4 [R|t] extrinsic (OpenCV-style block, last row [0,0,0,1])
        ext = np.eye(4)
        ext[:3,:3] = R_wc
        ext[:3, 3] = t
        
        d[2] += 0.001
        C = C0 + d

        poses.append(CameraInfo(uid=idx, R=R, T=C0, FovY=cam.FovY, FovX=cam.FovX, image=None,verts=cam.verts,
                        image_path=cam.image_path, image_name=cam.image_name, width=cam.width, height=cam.height,
                        time=cam.time, mask=None, feature=cam.feature))
                     

    return poses

def generateXYZfromDepth(depth_path, R, T,fovx, fovy):
    # Load depth map (assumed grayscale, single channel)
    depth_img = Image.open(depth_path) # "F" = 32-bit float
    scale = 0.25  # <-- change this factor as needed
    w, h = depth_img.size
    new_w, new_h = int(w * scale), int(h * scale)
    depth_img = depth_img.resize((new_w, new_h), Image.LANCZOS).convert("F") 
    
    depth = np.array(depth_img, dtype=np.float32)/255.   # HxW

    W,H = new_w, new_h 

    # Intrinsics from FOV
    fx = W / (2 * np.tan(fovx / 2))
    fy = H / (2 * np.tan(fovy / 2))
    cx, cy = W / 2, H / 2

    # Pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Camera coords
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)[::2]

def readHomeStudioInfo_singleview(path):
    print("Reading Training Transforms")
    # For now we are using single image so lets use a single set of vertices
    A = (458, 622)
    B = (3819, 511)
    C = (3999,2504)
    D = (375, 2554)
    verts = [A,B,C,D]
    
    
    train_cam_infos, R, T, fx, fy = generateSingleViewCameras(path, verts)
    video_cam_infos = circle_in_image_plane(R=R, C0=T, cam_infos=train_cam_infos, r=0.01, n_steps=70)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    xyz = generateXYZfromDepth('/home/barry/Desktop/other_code/Depth-Anything-V2/depth_vis/002.png', R, T, fx, fy)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))


    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        video_cameras=video_cam_infos,
        nerf_normalization=nerf_normalization,
    )
    return scene_info

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
    pts = pcd[::3, :3]
    col = pcd[::3, 3:]

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

sceneLoadTypeCallbacks = {
    "homestudio": readHomeStudioInfo
}
