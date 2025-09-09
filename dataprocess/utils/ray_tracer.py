import torch
import torch.nn as nn

def fov2_f_c(fovx, fovy, H, W):
    fx = 0.5 * W / torch.tan(0.5 * fovx)
    fy = 0.5 * H / torch.tan(0.5 * fovy)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    
    return fx, fy, cx, cy

class SharedIntrinsics(nn.Module):
    
    def __init__(self, H, W, fovx, fovy, device="cuda"):
        super().__init__()
        self.fov = nn.Parameter(torch.tensor([fovx, fovy]).to(device))

        self.H=H
        self.W=W
    
    def forward(self):
        return self.H,self.W, self.fov[0], self.fov[1]
        
class CameraTracer(nn.Module):
    def __init__(self, c2w,mask, image, device="cude"):
        super().__init__()
        
        self.device = device
        self.c2w = nn.Parameter(c2w.float())
        
        self.mask=mask
        self.image=image

    def forward(self, H, W, fovx, fovy):
        return self.generate_rays(H, W, fovx, fovy)
    
        
    def generate_rays(self, H, W, fovx, fovy):
        c2w_R = self.c2w[:3, :3]
        c2w_T = self.c2w[:3, 3]
        
        #Create pixel grid in NDC space
        i = torch.linspace(0, W-1, W, device=self.device, dtype=torch.float)
        j = torch.linspace(0, H-1, H, device=self.device, dtype=torch.float)
        jj, ii = torch.meshgrid(j, i, indexing="ij")  # [H,W] each

        # Get intrinsics from FOV (pinhole model)
        fx, fy, cx, cy = fov2_f_c(fovx, fovy, H, W)

        # Direction in camera space
        x = (ii - cx) / fx
        y = -(jj - cy) / fy   # flip y so +y is up
        z = -torch.ones_like(x, dtype=torch.float)  # camera looks down -z
        dirs_cam = torch.stack([x, y, z], dim=-1)  # [H,W,3]
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

        dirs_world = dirs_cam @ c2w_R.T  # [H,W,3]
        dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

        origins_world = c2w_T.expand_as(dirs_world)  # all rays from same origin

        return origins_world, -dirs_world



import torch.nn.functional as F
class QuadSurface(nn.Module):
    def __init__(self, abcd, H, W, texture, device="cuda"):
        super().__init__()
        self.abc = nn.Parameter(abcd[[0,1,3]].to(device))
        self.H, self.W = H, W
        
        self.texture = texture.to(device)
        self.device = device
    def forward(self, origin, direction):
        H, W = origin.shape[:2]
        origin = origin.reshape(-1, 3)
        direction = direction.reshape(-1, 3)

        a,b,c = self.abc
        # Calculate t intersection for each ray
        u = b-a
        v = c-a
        suf_normal = torch.cross(u,v, dim=-1).unsqueeze(-1)
        nom = torch.matmul(a.unsqueeze(0) - origin, suf_normal)
        denom = torch.matmul(direction, suf_normal)+1e-8
        t = nom/denom
        
        x = origin + t*direction
        # determine the normalized position w.r.t plane
        u_len2 = (u * u).sum()
        v_len2 = (v * v).sum()
        
        d = x - a  # (N,3)

        # projections (N,)
        u_coord = (d @ u) / u_len2
        v_coord = (d @ v) / v_len2
        
        uv = torch.stack([u_coord, v_coord], dim=-1)
        uv_mask = (uv[:,0] >= 0. ) &  (uv[:,0] <= 1. ) &  (uv[:,1] >= 0. ) &  (uv[:,1] <= 1. )
        uv = (2.* uv - 1.).unsqueeze(0).unsqueeze(2)
        # exit()
        sampled = F.grid_sample(self.texture.unsqueeze(0), uv, align_corners=True, mode="bilinear").squeeze(-1).squeeze(0)
        
        uv_mask = uv_mask.reshape(H, W )
        sampled = sampled.reshape(3, H, W )#.detach().cpu().numpy()
        
        return sampled
        # import matplotlib.pyplot as plt
        
        # plt.figure()
        # plt.imshow(sampled)
        # plt.show()
        # exit()

