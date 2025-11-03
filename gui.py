
import shutil
try:
    import dearpygui.dearpygui as dpg
except:
    print("No dpg running")
    dpg = None

import numpy as np
import random
import os, sys
import torch
import torchvision.utils as vutils

import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr, mse, rgb_to_ycbcr
from gaussian_renderer import render_extended




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
                 view_test,
                 use_gui:bool=False,
                 ):
        self.stage = 'fine'
        expname = 'output/'+expname
        self.expname = expname
        self.opt = opt
        self.pipe = pipe
        self.dataset = dataset
        self.dataset.model_path = expname
        
        # Metrics, Test images and Video Renders folders
        self.statistics_path = os.path.join(expname, 'statistics')
        self.save_tests = os.path.join(expname, 'tests')
        self.save_videos = os.path.join(expname, 'videos')
        

        self.hyperparams = hyperparams
        self.args = args
        self.args.model_path = expname
        self.saving_iterations = saving_iterations
        self.test_every = testing_iterations
        
        self.checkpoint = ckpt_start
        self.debug_from = debug_from
        
        self.results_dir = os.path.join(self.args.model_path, 'active_results')
        if ckpt_start is None:
            if not os.path.exists(self.args.model_path):os.makedirs(self.args.model_path)   

            if os.path.exists(self.results_dir):
                print(f'[Removing old results] : {self.results_dir}')
                shutil.rmtree(self.results_dir)
            os.mkdir(self.results_dir)
            
        if os.path.exists(self.statistics_path) == False:
            os.makedirs(self.statistics_path)
        if os.path.exists(self.save_tests) == False:
            os.makedirs(self.save_tests)
        if os.path.exists(self.save_videos) == False:
            os.makedirs(self.save_videos)
            
        # Set the background color
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Set the gaussian mdel and scene
        gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        if ckpt_start is not None:
            scene = Scene(dataset, gaussians, self.opt, args.cam_config, load_iteration=ckpt_start)
        else:

            scene = Scene(dataset, gaussians, self.opt, args.cam_config)
        
        # Initialize DPG      
        super().__init__(use_gui, scene, gaussians, self.expname, view_test)

        # Initialize training
        self.timer = Timer()
        self.timer.start()
        self.init_taining()
        
        if ckpt_start: 
            self.iteration = int(self.scene.loaded_iter) + 1
         
    def init_taining(self):
        # Set start and end of training
        self.final_iter = self.opt.iterations
            
        first_iter = 1

        self.gaussians.training_setup(self.opt)

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
            self.viewpoint_stack = self.scene.train_camera
            self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=16, collate_fn=list))

    @property
    def get_batch_views(self): 
        try:
            viewpoint_cams = next(self.loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=16, collate_fn=list, persistent_workers=False, pin_memory=False,)
            self.loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.loader)
        return viewpoint_cams


    def train_step(self, viewpoint_cams):
        """Training
        
        Notes:
            As we follow, c' = c + delta-c_i, and it is assumed c has been over-fit on the un-lit training set,
            We begin to also train the deformation functionality, leaving only the deformation method in the computational graph
            I.e. We "detach" the canonical parameters from this stage and backprop the loss based on the learned deformation
            
            The deformation follows the process of:
                1. Sampling a tri-plane grid to retrive continuous indexes (0<a<1 and 0<b<1)
                2. Scaling a,b to the backgground (cropped) image
                3. Mip-map sampling (again using torch grid_sampling)
                4. Processing the new color as 
                    c' = l.c + (1-l).c_mipmap
        """
        self.gaussians.pre_train_step(self.iteration, self.opt.iterations, 'fine')

        # Sample the background image
        textures = []
        for cam in viewpoint_cams:
            id1 = int(cam.time*self.scene.maxframes)
            textures.append(self.scene.ibl[id1])
        
        render, canon, info = render_extended(
            viewpoint_cams, 
            self.gaussians,
            textures,
            return_canon=True
        )

        self.gaussians.pre_backward(self.iteration, info)

        # Fetch and load training data
        render_gt = torch.cat([cam.image.unsqueeze(0) for cam in viewpoint_cams], dim=0).cuda()
        masked_gt = torch.cat([cam.sceneoccluded_mask.unsqueeze(0) for cam in viewpoint_cams], dim=0).cuda()
        canons_gt = torch.cat([cam.canon.unsqueeze(0) for cam in viewpoint_cams], dim=0).cuda()
        gt_out = render_gt* masked_gt
        canon_out = canons_gt* masked_gt
        
        # Loss Functions
        deform_loss = l1_loss(render, gt_out)
        canon_loss = l1_loss(canon, canon_out)

        
        
        
        dssim = (1-ssim(render, gt_out))/2.
        loss = (1-self.opt.lambda_dssim)*deform_loss + self.opt.lambda_dssim*dssim + self.opt.lambda_canon*canon_loss
                   
        # print( planeloss ,depthloss,hopacloss ,wopacloss ,normloss ,pg_loss,covloss)
        with torch.no_grad():
            if self.gui:
                dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                
                dpg.set_value("_log_relit", f"Relit Loss: {deform_loss.item()}")
                dpg.set_value("_log_canon", f"ssim {dssim.item():.5f} | canon {canon_loss.item():.5f}")
                # dpg.set_value("_log_deform", f"residuals {residual_loss.item():.5f}")
                dpg.set_value("_log_points", f"Point Count: {self.gaussians.get_xyz.shape[0]}")

            
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                    
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()

        self.gaussians.post_backward(self.iteration, info, 'fine')

    @torch.no_grad
    def test_step(self, viewpoint_cams, index, d_type):
        # Sample the background image
        id1 = int(viewpoint_cams.time*self.scene.maxframes)
        texture = self.scene.ibl[id1].cuda()
        # Rendering pass
        relit, _ = render_extended(
            [viewpoint_cams], 
            self.gaussians,
            [texture],
        )
        
        # Process data
        relit = relit.squeeze(0)

        mask = viewpoint_cams.sceneoccluded_mask.cuda()
        gt_img = viewpoint_cams.image.cuda() #* (viewpoint_cams.sceneoccluded_mask.cuda())
        
        gt_out = gt_img * mask

        # Save image
        # if self.iteration > (self.final_iter - 500) or  index % 5 == 0:
        #     save_im = mask*relit + (1.-mask)*gt_img
        #     vutils.save_image(save_im, os.path.join(self.save_tests, f"{d_type}_{index:05}.jpg"))

        
        gt_ycc = rgb_to_ycbcr(gt_out).squeeze(0)
        relit_ycc = rgb_to_ycbcr(relit).squeeze(0)
    
        
        return {
            "mse":mse(relit, gt_out),
            "psnr":psnr(relit, gt_out),
            "psnr-y":psnr(relit_ycc[0, ...], gt_ycc[0, ...]),
            "psnr-crcb":psnr(relit_ycc[1:, ...], gt_ycc[1:, ...]),
            "ssim":ssim(relit, gt_out)
        }
        
    @torch.no_grad
    def video_step(self, viewpoint_cams, index):
        # Sample the background image
        texture = self.scene.ibl[index%99].cuda()
        # Rendering pass
        _, relit, _ = render_extended(
            [viewpoint_cams], 
            self.gaussians,
            [texture],return_canon=True
        )
        
        # Process data
        relit = relit.squeeze(0)
        
        vutils.save_image(relit, os.path.join(self.save_videos, f"{index:05}.jpg"))


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()

    # print('Runing from ... ',os.environ["SLURM_PROCID"])
    # exit()
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
    parser.add_argument('--view-test', action='store_true', default=False)
    
    parser.add_argument("--cam-config", type=str, default = "4")
    parser.add_argument("--downsample", type=int, default=1)
    
    
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
        view_test=args.view_test,

        use_gui=False if dpg is None else True,
    )
    gui.render()
    del gui
    torch.cuda.empty_cache()
