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

# AbsGS for better split on larger points split from https://github.com/TY424/AbsGS/blob/main/scene/gaussian_model.py

import torch
import numpy as np
from utils.general_utils import get_expon_lr_func
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid

        self.sigmoid_activation = torch.sigmoid
        
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._deformation = deform_network(args)
        
        self.gsplat_optimizers = None
        self.hex_optimizer = None
        
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.spatial_lr_scale_background = 0
        self.target_neighbours = None
        
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._deformation.state_dict(),
            self.splats,
            self.hex_optimizer.state_dict(),
            self.spatial_lr_scale,
            self.spatial_lr_scale_background,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        deform_state,
        self.splats,
        opt_dict,
        self.spatial_lr_scale,self.spatial_lr_scale_background) = model_args
        
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)

        self.hex_optimizer.load_state_dict(opt_dict)


    @property
    def get_features(self):
        features_dc = self.splats['sh0']
        features_rest = self.splats['shN']
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.splats["scales"])
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.splats["quats"])

    @property
    def get_xyz(self):
        return self.splats["means"]

    @property
    def get_opacity(self):
        return torch.sigmoid(self.splats["opacities"])
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    @property
    def get_covmat(self):
        w, x, y, z = self.get_rotation.unbind(-1)
        scale = self.get_scaling
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = torch.stack([
            torch.stack([1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)], dim=-1),
        ], dim=-2)
        
        e1 = torch.tensor([1,0,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)
        e2 = torch.tensor([0,1,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)

        # Scale local basis
        v1 = e1 * scale[:, [0]]  # (N,3)
        v2 = e2 * scale[:, [1]]  # (N,3)

        # Apply rotation: batch matmul (N,3,3) @ (N,3,1) -> (N,3,1)
        t_u = torch.bmm(R, v1.unsqueeze(-1)).squeeze(-1)  # (N,3)
        t_v = torch.bmm(R, v2.unsqueeze(-1)).squeeze(-1)  # (N,3)

        # Magnitudes
        m_u = torch.linalg.norm(t_u, dim=-1)
        m_v = torch.linalg.norm(t_v, dim=-1)

        # Directions (normalized)
        d_u = t_u / m_u.unsqueeze(-1)
        d_v = t_v / m_v.unsqueeze(-1)

        magnitudes = torch.stack([m_u, m_v], dim=-1)     # (N,2)
        directions = torch.stack([d_u, d_v], dim=1)      # (N,2,3)

        return magnitudes, directions

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):        
        ##### Set-up GSplat optimizers #####
        self.percent_dense = training_args.percent_dense
        
        from gsplat import DefaultStrategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=0.001,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            
            prune_scale3d=0.1,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=training_args.densify_from_iter,
            refine_stop_iter=training_args.densify_until_iter,
            reset_every=training_args.opacity_reset_interval,
            refine_every=training_args.densification_interval,
            absgrad=False,
            revised_opacity=False,
            key_for_gradient="gradient_2dgs",
        )
        self.strategy.check_sanity(self.splats, self.gsplat_optimizers)
        self.strategy_state = self.strategy.initialize_state()
        
        
        ##### Set-up hex-plane optimizers #####
        self.hex_optimizer = torch.optim.Adam(
            [        
                {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
                {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            ],
            lr=0.0, eps=1e-15
        )

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def pre_train_step(self, iteration, max_iterations):
        """Run pre-training step functions"""
        if iteration < max_iterations:
            self.hex_optimizer.zero_grad(set_to_none=True)
        
        # For hex-plane parameters
        for param_group in self.hex_optimizer.param_groups:
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr

        if iteration % 500 == 0:
            self.oneupSHdegree()
        return None    

    def pre_backward(self, iteration, info):
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.gsplat_optimizers,
            state=self.strategy_state,
            step=iteration,
            info=info,
        )
        
    def post_backward(self, iteration, info):
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.gsplat_optimizers,
            state=self.strategy_state,
            step=iteration,
            info=info,
            packed=False,
        )
        for optimizer in self.gsplat_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        self.hex_optimizer.step()
        self.hex_optimizer.zero_grad(set_to_none=True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self.splats["opacities"].shape[1]):
            l.append('opacity_{}'.format(i))
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))

        return l
    
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.splats["means"].detach().cpu().numpy()
        opacities = self.splats["opacities"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()
        
        f_dc = self.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        

        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, colors, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, training_args):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opac_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity")]
        opac_names = sorted(opac_names, key = lambda x: int(x.split('_')[-1]))
        opacities = np.zeros((xyz.shape[0], len(opac_names)))
        for idx, attr_name in enumerate(opac_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        col_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color")]
        col_names = sorted(col_names, key = lambda x: int(x.split('_')[-1]))
        cols = np.zeros((xyz.shape[0], len(col_names)))
        for idx, attr_name in enumerate(col_names):
            cols[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        
        self.active_sh_degree = self.max_sh_degree
        
        self.params = {
            ("means", nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.position_lr_init * self.spatial_lr_scale),
            ("scales", nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.scaling_lr),
            ("quats", nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.rotation_lr),
            ("opacities",nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.opacity_lr),
            ("sh0", nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)), training_args.feature_lr),
            ("shN", nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)), training_args.feature_lr/20.),
            
        }
        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in self.params}).cuda()
        
        import math
        batch_size = training_args.batch_size
        self.gsplat_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(batch_size)}],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in self.params
        }
    
    def create_from_pcd(self, pcd : BasicPointCloud, training_args):
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        xyz_min = fused_point_cloud.min(0).values
        xyz_max = fused_point_cloud.max(0).values
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
    
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Initialize opacities
        opacities = 0.95 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        
        mean_foreground = fused_point_cloud.mean(dim=0).unsqueeze(0)
        dist_foreground = torch.norm(fused_point_cloud - mean_foreground, dim=1)
        self.spatial_lr_scale = torch.max(dist_foreground).detach().cpu().numpy()
        
        print(f"Target lr scale: {self.spatial_lr_scale}")
        self.active_sh_degree = 0
        
        # initialize the splats
        params = {
            # 2DGS/3DGS Parameters
            ("means", nn.Parameter(fused_point_cloud.requires_grad_(True)), training_args.position_lr_init * self.spatial_lr_scale),
            ("scales", nn.Parameter(scales.requires_grad_(True)), training_args.scaling_lr),
            ("quats", nn.Parameter(rots.requires_grad_(True)), training_args.rotation_lr),
            ("opacities", nn.Parameter(opacities.requires_grad_(True)), training_args.opacity_lr),
            ("sh0", nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)), training_args.feature_lr),
            ("shN", nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)), training_args.feature_lr/20.),
            
        }
        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to("cuda")

        import math
        batch_size = 1 #training_args.batch_size
        self.gsplat_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(batch_size)}],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        }
        

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight,
                           minview_weight):
        tvtotal = 0
        l1total = 0
        tstotal = 0
        col=0
        
        wavelets = self._deformation.deformation_net.grid.waveplanes_list()
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for index, grids in enumerate(self._deformation.deformation_net.grid.grids_()):
            if index in [0,1,3]: # space only
                for grid in grids:
                    tvtotal += compute_plane_smoothness(grid)
            elif index in [2, 4, 5]:
                for grid in grids: # space time
                    tstotal += compute_plane_smoothness(grid)
                
                for grid in wavelets[index]:
                    l1total += torch.abs(grid).mean()    
        
        return plane_tv_weight * tvtotal + time_smoothness_weight*tstotal + l1_time_planes_weight*l1total


from scipy.spatial import KDTree
import torch

def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)