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

import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset,IBLBackround
from arguments import ModelParams
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, num_cams='4', load_iteration=None, skip_coarse=None, max_frames=50):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        max_frames = 4
        num_cams = 4
        scene_info = sceneLoadTypeCallbacks["homestudio"](args.source_path)
        dataset_type="condense"
        
        self.maxframes = max_frames
        self.num_cams = num_cams
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        self.train_camera = FourDGSdataset(scene_info.train_cameras, dataset_type)
        self.video_cameras = FourDGSdataset(scene_info.video_cameras, dataset_type)
        self.ibl = IBLBackround(scene_info.background_pth_ids)
        
        self.point_cloud = scene_info.point_cloud
        
        if self.loaded_iter:
            print(f'Load from iter {self.loaded_iter}')

            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud) # TODO: configure

    def save(self, iteration, stage):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def update_scene(self, abc, cams):
        # Note that this needs to be run before the init_training function (where we set-up the mask flags)
        #   so that the c2w and intrinsics get updated for training
        # Update training camera parameters and scene abc
        self.ibl.update_abc(abc)
        
        from scene.dataset_readers import CameraInfo
        
        caminfo = []
        for cam in cams:
            caminfo.append(
                
                CameraInfo(
                    uid=cam.uid,
                    R = cam.R,
                    T = cam.T,
                    fx=cam.fx, fy=cam.fy,
                    cx=cam.cx, cy=cam.cy,
                    height=cam.image_height, width=cam.image_width,
                    
                    image_path=cam.image_path,
                    
                    s_path=cam.scene_path,
                    so_path=cam.sceneoccluded_path,
                    g_path=cam.glass_path,
                    b_path=None,


                    time=cam.time,
                    feature=None
                    
                )
            )
        self.train_camera = FourDGSdataset(caminfo, self.dataset_type)
        

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def index_train(self, index):
        return self.train_camera[index]
    