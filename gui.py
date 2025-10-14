
import shutil
import dearpygui.dearpygui as dpg
import numpy as np
import random
import os, sys
import torch

import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer

from utils.loss_utils import l1_loss

from gaussian_renderer import render_batch, render, render_extended

from torch.cuda.amp import autocast, GradScaler

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
import matplotlib.pyplot as plt

from gui_utils.base import GUIBase
class GUI(GUIBase):
    def __init__(self, 
                 args, 
                 hyperparams, 
                 dataset, 
                 opt, 
                 pipe, 
                 testing_iterations, 
                 saving_iterations,
                 ckpt_start,
                 debug_from,
                 expname,
                 skip_coarse,
                 view_test,
                 bundle_adjust,
                 use_gui:bool=False
                 ):

        self.skip_coarse = None
        self.stage = 'coarse'

        expname = 'output/'+expname
        self.expname = expname
        self.opt = opt
        self.pipe = pipe
        self.dataset = dataset
        self.dataset.model_path = expname
        self.hyperparams = hyperparams
        self.args = args
        self.args.model_path = expname
        self.saving_iterations = saving_iterations
        self.checkpoint = ckpt_start
        self.debug_from = debug_from

        self.total_frames = 300
        
        self.results_dir = os.path.join(self.args.model_path, 'active_results')
        if ckpt_start is None:
            if not os.path.exists(self.args.model_path):os.makedirs(self.args.model_path)   

            if os.path.exists(self.results_dir):
                print(f'[Removing old results] : {self.results_dir}')
                shutil.rmtree(self.results_dir)
            os.mkdir(self.results_dir)    
            
        # Set the background color
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Set the gaussian mdel and scene
        gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        if ckpt_start is not None:
            scene = Scene(dataset, gaussians, self.opt, args.cam_config, load_iteration=ckpt_start)
        else:
            if skip_coarse:
                gaussians.active_sh_degree = dataset.sh_degree
            scene = Scene(dataset, gaussians, self.opt, args.cam_config, skip_coarse=self.skip_coarse)
        
        self.total_frames = scene.maxframes
        # Initialize DPG      
        super().__init__(use_gui, scene, gaussians, self.expname, view_test, bundle_adjust)

        
        # Initialize training
        self.timer = Timer()
        self.timer.start()
        self.init_taining()
        
        if skip_coarse:
            self.iteration = 1
        if ckpt_start: self.iteration = int(self.scene.loaded_iter) + 1
         
    def init_taining(self):
        self.final_iter = self.opt.iterations
        first_iter = 1

        # Set up gaussian training
        self.gaussians.training_setup(self.opt)
        # Load from fine model if it exists

        if self.checkpoint:
            (model_params, first_iter) = torch.load(f'{self.expname}/chkpnt_{self.checkpoint}.pth')
            self.gaussians.restore(model_params, self.opt)

        # Set current iteration
        self.iteration = first_iter

        # Events for counting duration of step
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        if self.view_test == False:
            self.random_loader  = True

            print('Loading dataset..')
            # Set the data we want to use from this dataset
            self.scene.train_camera.loading_flags["image"] = True
            self.scene.train_camera.loading_flags["scene_occluded"] = True


            self.viewpoint_stack = self.scene.train_camera
            self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=4, collate_fn=list))
    @property
    def get_batch_views(self, stack=None):
        self.scene.train_camera.loading_flags["image"] = True
        self.scene.train_camera.loading_flags["scene_occluded"] = True
            
        try:
            viewpoint_cams = next(self.loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=4, collate_fn=list, persistent_workers=False, pin_memory=False,)
            self.loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.loader)
        
        return viewpoint_cams

    def train_coarse_step(self, viewpoint_cams):
        
        self.gaussians.pre_train_step(self.iteration, self.opt.iterations)

       
        rgb, info, normal_err, alpha_err = render_batch(
            viewpoint_cams, 
            self.gaussians,
        )
        
        self.gaussians.pre_backward(self.iteration, info)
        
        gt_img = viewpoint_cams.image.cuda() * (viewpoint_cams.sceneoccluded_mask.cuda())
        
        loss = l1_loss(rgb, gt_img) +  self.opt.lambda_alpha * alpha_err #self.opt.lambda_normal * normal_err +
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    dpg.set_value("_log_loss", f"Loss: {loss.item()}")
                    dpg.set_value("_log_depth", f"Norm Loss: {self.opt.lambda_normal * normal_err.item()}")
                    dpg.set_value("_log_opacs", f"Alpha/Mask Loss: {self.opt.lambda_alpha * alpha_err.item()}")
                    # dpg.set_value("_log_dynscales", f"Inv Loss: {self.opt.lambda_inv * inv_err.item()}")
                    dpg.set_value("_log_points", f"Number points: {self.gaussians.get_xyz.shape[0]} ")

            
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                    
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()

        self.gaussians.post_backward(self.iteration, info)

    
    def train_step(self, viewpoint_cams):
        
        self.gaussians.pre_train_step(self.iteration, self.opt.iterations)
        
        abc = self.scene.ibl.abc.cuda()
                
        id1 = int(viewpoint_cams.time*100)
        texture = self.scene.ibl[id1].cuda()
        
        rgb, info = render_extended(
            viewpoint_cams, 
            self.gaussians,
            abc,
            texture,
            self.optix_runner
        )
        
        self.gaussians.pre_backward(self.iteration, info)
        
        gt_img = viewpoint_cams.image.cuda() * (viewpoint_cams.sceneoccluded_mask.cuda())

        loss = l1_loss(rgb, gt_img)

        # planeloss = self.gaussians.compute_regulation(
        #     self.hyperparams.time_smoothness_weight, self.hyperparams.l1_time_planes, self.hyperparams.plane_tv_weight,
        #     self.hyperparams.minview_weight
        # )
                   
        # print( planeloss ,depthloss,hopacloss ,wopacloss ,normloss ,pg_loss,covloss)
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    dpg.set_value("_log_loss", f"Loss: {loss.item()}")
                    # dpg.set_value("_log_depth", f"Depth Loss: {dL1.item()}")
                    dpg.set_value("_log_points", f"Number points: {self.gaussians.get_xyz.shape[0]} ")

            
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                    
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()

        self.gaussians.post_backward(self.iteration, info)

        

    def nvs(self):

        # Start recording step duration
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        cameras = self.scene.video_cameras        
        
        for idx, cam in enumerate(cameras):
            buffer_image = render(
                    cam,
                    self.gaussians, 
                    view_args={
                        "vis_mode":"render"
                    }
            )["render"]
            
            save_novel_views(buffer_image, 
                idx, self.args.expname)
            
            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H,self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            buffer_image = self.buffer_image

            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!
            
            # Add _log_view_camera
            dpg.set_value("_log_view_camera", f"Rendering Nove Views")


            dpg.render_dearpygui_frame()

import cv2
def save_novel_views(pred, idx, name):
    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    pred = pred.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    if not os.path.exists(f'output/{name}/nvs/'):
        os.mkdir(f'output/{name}/nvs/')
        os.mkdir(f'output/{name}/nvs/full/')
    elif not os.path.exists(f'output/{name}/nvs/full/'):
        os.mkdir(f'output/{name}/nvs/full/')
    cv2.imwrite(f'output/{name}/nvs/full/{idx}.png', pred_bgr)

    return pred_bgr

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=4000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8000, 15999, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument('--skip-coarse', type=str, default = None)
    parser.add_argument('--view-test', action='store_true', default=False)
    
    parser.add_argument('--bundle-adjust', action='store_true', default=False)

    parser.add_argument("--cam-config", type=str, default = "4")
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
        
    
    torch.autograd.set_detect_anomaly(True)
    hyp = hp.extract(args)
    initial_name = args.expname     
    name = f'{initial_name}'
    gui = GUI(
        args=args, 
        hyperparams=hyp, 
        dataset=lp.extract(args), 
        opt=op.extract(args), 
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations, 
        saving_iterations=args.save_iterations,
        ckpt_start=args.start_checkpoint, 
        debug_from=args.debug_from, 
        expname=name,
        skip_coarse=args.skip_coarse,
        view_test=args.view_test,
        bundle_adjust=args.bundle_adjust,

        use_gui=True
    )
    gui.render()
    del gui
    torch.cuda.empty_cache()
    # TV Reg
    # hyp.plane_tv_weight = 0.
    # for value in [0.001,0.00075,0.00025,0.0001,]:
    #     name = f'{initial_name}_TV{value}'
    #     hyp.plane_tv_weight = value
        
    #     # Start GUI server, configure and run training
    #     gui = GUI(
    #         args=args, 
    #         hyperparams=hyp, 
    #         dataset=lp.extract(args), 
    #         opt=op.extract(args), 
    #         pipe=pp.extract(args),
    #         testing_iterations=args.test_iterations, 
    #         saving_iterations=args.save_iterations,
    #         ckpt_start=args.start_checkpoint, 
    #         debug_from=args.debug_from, 
    #         expname=name,
    #         skip_coarse=args.skip_coarse,
    #         view_test=args.view_test
    #     )