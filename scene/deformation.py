import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from scene.triplanes import TriPlaneField

import time

from torch_cluster import knn_graph
from utils.general_utils import strip_symmetric, build_scaling_rotation

# Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float).cuda()  # shape (3, 3)

XYZ2RGB = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=torch.float).cuda()  # shape (3, 3)


def rgb_to_xyz(rgb):
    threshold = 0.04045
    rgb_linear = torch.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    
    xyz = torch.matmul(rgb_linear, RGB2XYZ.T)

    return xyz

def xyz_to_rgb(xyz):
    rgb_linear = torch.matmul(xyz, XYZ2RGB.T)

    threshold = 0.0031308
    rgb =  torch.where(
        rgb_linear <= threshold,
        12.92 * rgb_linear,
        1.055 * (rgb_linear.clamp(min=1e-8) ** (1/2.4)) - 0.055
    )

    return rgb.clamp(0.0, 1.0) 

class Deformation(nn.Module):
    def __init__(self, W=256, args=None):
        super(Deformation, self).__init__()
        self.W = W
        self.grid = WavePlaneField(args.bounds, args.target_config)

        self.args = args

        self.ratio=0
        self.create_net()
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # inputs scaling, scalingmod=1.0, rotation
        self.covariance_activation = build_covariance_from_scaling_rotation

        
    def set_aabb(self, xyz_max, xyz_min, grid_type='target'):
        if grid_type=='target':
            self.grid.set_aabb(xyz_max, xyz_min)
  
    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        self.spacetime_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))

        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
    
    def query_spacetime(self, rays_pts_emb, time, covariances):

        space, spacetime, coltime = self.grid(rays_pts_emb[:,:3], time, covariances)

        st = self.spacetime_enc(space * spacetime)
        return st
    
    
    def forward(self,rays_pts_emb, rotations_emb, scale_emb, shs_emb, view_dir, time_emb, h_emb):

        covariances = self.covariance_activation(scale_emb, 1., rotations_emb)
        dyn_feature = self.query_spacetime(rays_pts_emb,time_emb, covariances)
        
        shs = shs_emb + self.shs_deform(dyn_feature).view(-1, 16, 3)
        
        # Essentially no view-dependancy
        # shs[1:] *= 0.
        
        return rays_pts_emb, rotations_emb, h_emb[:,0].unsqueeze(-1), shs, None
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list
    

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width

        self.deformation_net = Deformation(W=net_width,  args=args)

        self.apply(initialize_weights)

    def forward(self, point, rotations=None, scales=None, shs=None,view_dir=None, times_sel=None, h_emb=None, iteration=None):

        return  self.deformation_net(
            point,
            rotations,
            scales,
            shs,
            view_dir,
            times_sel, 
            h_emb=h_emb, 
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

    def get_dyn_coefs(self, xyz, scale):
        return self.deformation_net.get_dx_coeffs(xyz, scale)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
            