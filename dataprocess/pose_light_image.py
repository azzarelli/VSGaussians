from torchvision import transforms as T
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
import cv2
import os

from scene.dataset_readers import readHomeStudioInfo
from dataprocess.utils.ray_tracer import CameraTracer, SharedIntrinsics, QuadSurface
transform = T.ToTensor()
device = "cuda"

# Resolve the position and orientation of the plane
path_2_dataset = "/media/barry/56EA40DEEA40BBCD/DATA/studio_test2/"

scene_info = readHomeStudioInfo(path_2_dataset)
# Contains :
#   point_cloud : point cloud
#   train_cameras : list of CameraInfo class
#       b_path : for background path
#       s0_path : obstructed screen mask (alpha=0 is screen area)
#       H,W,FovX,FovY, .. : camera parameters

scene_cameras = scene_info.train_cameras


initial_mesh = nn.Parameter(torch.randn(4,3)) # A,B,C,D

# Initialize the camera object for bundle adjustment on pose and intrinsics
train_cameras = []
for cam  in scene_cameras:
    mask = Image.open(cam.so_path).split()[-1] # get alpha chamme;
    mask = (1.- transform(mask).cuda()).bool().squeeze(0)
    image = Image.open(cam.image_path).convert("RGB")
    image = transform(image).cuda() * mask
    
    train_cameras.append(CameraTracer(
        torch.from_numpy(cam.c2w).to(device), 
        mask, 
        image,
        device=device
    ))

light_image = Image.open(os.path.join(path_2_dataset, "cropped_images/image_001.png")).convert("RGB")
light_image = transform(light_image)

# Initilize the shared intrinsic parameters
intrinsics = SharedIntrinsics(
    cam.height, cam.width,
    cam.FovX, cam.FovY, device=device
)

# Initialize the A,B,C,D 3D texture coords
def furthest_point_on_ray(o, d, points):
    """
    o: (3,) ray origin
    d: (3,) ray direction
    points: (N, 3) point cloud
    """
    d = d / d.norm()

    # project points onto ray
    v = points - o[None, :]    # (N,3)
    t_vals = v @ d             # (N,)

    # keep only points in front
    valid = t_vals >= 0
    if not valid.any():
        return None, None

    t_vals = t_vals[valid]

    mode = "mean"
    if mode == "nearest":
        t = t_vals.min()
    elif mode == "furthest":
        t = t_vals.max()
    elif mode == "mean":
        t = t_vals.mean()
    elif mode == "median":
        t = t_vals.median()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pos = o + t * d
    return pos

abcd = []
pcd = torch.from_numpy(scene_info.point_cloud.points).to(device).float()
for cam , train_cam in zip(scene_cameras, train_cameras):
    so_mask = Image.open(cam.so_path).split()[-1] # get alpha chamme;
    so_mask = (1.- transform(so_mask)).bool().squeeze(0).numpy().astype(np.uint8)
    s_mask = Image.open(cam.s_path).split()[-1] # get alpha chamme;
    s_mask = (1.- transform(s_mask)).bool().squeeze(0).numpy().astype(np.uint8)
    
    masked_img = s_mask
    
    # Solve the vertices of each image
    contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    
    peri = cv2.arcLength(cnt, True)
    for eps in np.linspace(0.01, 0.1, 20):   # try different tolerances
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)   # (x, y)
            break
    else:    
        raise RuntimeError("Could not approximate quadrilateral")
    
    """ To display the A,B,C,D verts uncomment the following...
    import matplotlib.pyplot as plt
    mask_rgb = cv2.cvtColor(masked_img * 255, cv2.COLOR_GRAY2RGB)
    for (x, y) in quad:
        cv2.circle(mask_rgb, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)

    plt.imshow(mask_rgb)
    plt.axis("off")
    plt.show()
    """
    
    # Order the ABCD image vertices
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4,2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]       # top-left (min x+y)
    ordered[2] = pts[np.argmax(s)]       # bottom-right (max x+y)
    ordered[1] = pts[np.argmin(diff)]    # top-right (min x-y)
    ordered[3] = pts[np.argmax(diff)]    # bottom-left (max x-y)
    ordered = torch.from_numpy(ordered).int()
    
    # Now we need to project the verts into 3-D to sample the pcd for each vert
    o, d = train_cam(288, 512, intrinsics.fov[0], intrinsics.fov[1])
    ABCD = []
    for (x,y) in ordered:
        x,y = int(x/2.5), int(y/2.5)
        o_vert = o[y, x]
        d_vert = d[y, x]
        # Get the further point along each ray
        point = furthest_point_on_ray(o_vert, d_vert, pcd)
        ABCD.append(point.unsqueeze(0))
    abcd.append(torch.cat(ABCD, dim=0).unsqueeze(0))

abcd = torch.cat(abcd, dim=0)
abcd = abcd.mean(0) # take the mean/median on the point cloud positions    

surface = QuadSurface(abcd, light_image.shape[1], light_image.shape[2], light_image)

"""Training"""
"""
We have three learnable parameters:
    surface: QuadSurface, a class with property abcd containing a (4,3) representation of A,B,C,D vertices
    intrinsics: SharedIntrinsis, a class with H,W, and learnable fovx fovy paramets, that are called in that order using forward
    train_cameras: list of CameraTracer, a class containing mask, image and learnable c2w data. Ray origin and directions can be called with `generate_rays` 

The objective now it to learn the best pose, intrinsics and ABCD vertices for our scene
"""
intr_optim = torch.optim.Adam(intrinsics.parameters(), lr=1e-3)
cam_optims = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in train_cameras]
suf_optim = torch.optim.Adam(surface.parameters(), lr=1e-3)

max_iterations = 10000
for iteration in range(max_iterations):
    intr_optim.zero_grad()
    suf_optim.zero_grad()
    for opt in cam_optims:
        opt.zero_grad()
    loss = 0.
    # Loop through each training camera
    for cam in train_cameras:
        # Get shared intrinsics
        H,W,fovx,fovy = intrinsics()
        
        # Get the origin and direction of each ray
        o,d = cam(H,W,fovx,fovy)
        
        # Get each surface
        render = surface(o, d)
        
        loss += ((cam.image - render)**2).mean()
        
    loss.backward()
    intr_optim.step()
    suf_optim.step()
    for opt in cam_optims:
        opt.step()
        
    if iteration % 100 ==0: print(f"{iteration}: {loss}")
        # exit()
        # print(cam.image.shape, render.shape)

# Showcase
import matplotlib.pyplot as plt
print("Results: ")
print(f"ABC Surface: {surface.abc}")
for idx, cam in enumerate(train_cameras):
    print(f"For Cam {idx}")
    # Get shared intrinsics
    H,W,fovx,fovy = intrinsics()
    print(H,W,fovx, fovy)
    # Get the origin and direction of each ray
    o,d = cam(H,W,fovx,fovy)
    print(cam.c2w)
    # Get each surface
    render = surface(o, d)
    
    img = torch.cat([cam.image, render], dim=-1).permute(1,2,0).detach().cpu().numpy()
    
    # plt.figure()
    # plt.imshow(img)
    # plt.show()