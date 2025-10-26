import dearpygui.dearpygui as dpg
import numpy as np
import os
import copy
import psutil
import torch
from gaussian_renderer import render, render_triangles
from tqdm import tqdm
import time
import json

class GUIBase:
    """This method servers to intialize the DPG visualization (keeping my code cleeeean!)
    
        Notes:
            none yet...
    """
    def __init__(self, gui, scene, gaussians, runname, view_test):
        
        self.gui = gui
        self.scene = scene
        self.gaussians = gaussians
        self.runname = runname
        self.view_test = view_test
        
        # Set the width and height of the expected image
        self.W, self.H = self.scene.train_camera[0].image_width, self.scene.train_camera[0].image_height

        if self.H > 1200 and self.scene != "dynerf":
            self.W = self.W//2
            self.H = self.H //2
        # Initialize the image buffer
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        
        # Other important visualization parameters
        self.time = 0.
        self.show_radius = 30.
        self.vis_mode = 'render'
        self.show_mask = 'none'
        self.show_dynamic = 0.
        self.w_thresh = 0.
        self.h_thresh = 0.
        
        self.set_w_flag = False
        self.w_val = 0.01
        # Set-up the camera for visualization
        self.show_scene_target = 0

        self.finecoarse_flag = True        
        self.switch_off_viewer = False
        self.switch_off_viewer_args = False
        self.full_opacity = False
        
        self.N_pseudo = 3 
        self.free_cams = [cam for idx, cam in enumerate(self.scene.test_camera) if idx % 10 == 0] 
        
        self.current_cam_index = 0
        self.original_cams = [copy.deepcopy(cam) for cam in self.free_cams]
        
        self.trainable_abc = None
        if self.gui:
            print('DPG loading ...')
            dpg.create_context()
            self.register_dpg()
            

    def __del__(self):
        dpg.destroy_context()

    def track_cpu_gpu_usage(self, time):
        # Print GPU and CPU memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)  # Convert to MB

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(
            f'[{self.stage} {self.iteration}] Time: {time:.2f} | Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB | CPU Memory Usage: {memory_mb:.2f} MB')
    
    def render(self):

        while dpg.is_dearpygui_running():
            if self.view_test == False:
                dpg.set_value("_log_stage", self.stage)

                if self.iteration <= self.final_iter:
                    
                    xyz_start = self.gaussians.get_xyz.shape[0]
                    # Get batch data
                    viewpoint_cams = self.get_batch_views
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.iter_start.record()
                    
                    # Depending on stage process a training step
                    self.train_step(viewpoint_cams)

                    self.iter_end.record()
                
                else: # Initialize fine from coarse stage
                    if self.stage == 'fine':
                        self.stage = 'done'
                        dpg.stop_dearpygui()
                        
                xyz_end = self.gaussians.get_xyz.shape[0]
                
                if xyz_start != xyz_end or self.iteration % 500 == 0:
                    self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)

                
                # Test Step
                if self.iteration % self.test_every == 0:
                    metrics = {
                        "mse":0.,
                        "psnr":0.,
                        "ssim":0.
                    }
                    
                    datasets = {
                        "L":metrics.copy(),
                        "V":metrics.copy(),
                        "LV":metrics.copy(),
                    }
                    test_size = len(self.scene.test_camera)
                    dataset_idxs = self.scene.test_camera.subset_idxs

                    for i, test_cam in enumerate(self.scene.test_camera):
                        metric_results = self.test_step(test_cam)
                        
                        if i < dataset_idxs[0]: # L-only tests
                            d_type = "L"
                        elif i < dataset_idxs[0] + dataset_idxs[1]: # V-only test
                            d_type = "V"
                        else:
                            d_type = "LV"
                            
                        for key in metrics.keys():
                            datasets[d_type][key] += metric_results[key].item()
                        
                        dpg.set_value("_log_test_progress", f"{int(100*(i/test_size))}% | {metric_results['psnr']:.2f} on {d_type} type")
                        dpg.render_dearpygui_frame()

                    # Average
                    for key, lengths in zip(datasets.keys(), dataset_idxs):
                        for key_1 in metrics.keys():
                            datasets[key][key_1] /= lengths

                    # Logs with 3 decimal places
                    dpg.set_value("_log_l_1", f"mse  : {datasets['L']['mse']:.3f}")
                    dpg.set_value("_log_l_2", f"ssim : {datasets['L']['ssim']:.3f}")
                    dpg.set_value("_log_l_3", f"psnr : {datasets['L']['psnr']:.2f}")

                    dpg.set_value("_log_v_1", f"mse  : {datasets['V']['mse']:.3f}")
                    dpg.set_value("_log_v_2", f"ssim : {datasets['V']['ssim']:.3f}")
                    dpg.set_value("_log_v_3", f"psnr : {datasets['V']['psnr']:.2f}")

                    dpg.set_value("_log_lv_1", f"mse  : {datasets['LV']['mse']:.3f}")
                    dpg.set_value("_log_lv_2", f"ssim : {datasets['LV']['ssim']:.3f}")
                    dpg.set_value("_log_lv_3", f"psnr : {datasets['LV']['psnr']:.2f}")
                    dpg.set_value("_log_test_progress", f"Saving json ...")
                    dpg.render_dearpygui_frame()
                    
                    test_fp = os.path.join(self.statistics_path, f"metrics_{self.iteration}.json")
                    with open(test_fp, "w") as outfile:
                        json.dump(datasets, outfile, indent=4, ensure_ascii=False)
                        
                        
                    dpg.set_value("_log_test_progress", f"...(training)...")
                    dpg.render_dearpygui_frame()
                    
                    
                # Update iteration
                self.iteration += 1


            with torch.no_grad():
                self.viewer_step()
                dpg.render_dearpygui_frame()    
                
            with torch.no_grad():
                self.timer.pause() # log and save
                torch.cuda.synchronize()
                if self.iteration % 1000 == 500: # make it 500 so that we dont run this while loading view-test
                    self.track_cpu_gpu_usage(0.1)
                    
                # Save scene when at the saving iteration
                if self.stage == 'fine' and ((self.iteration in self.saving_iterations) or (self.iteration == self.final_iter-1)):
                    self.save_scene()

                self.timer.start()
                
        dpg.destroy_context()
 
    @torch.no_grad()
    def viewer_step(self):
        t0 = time.time()
        if self.switch_off_viewer == False:

            cam = self.free_cams[self.current_cam_index]
            cam.time = self.time
            
            
            if self.trainable_abc is None:
                abc = self.scene.ibl.abc.cuda()
                
                id1 = int(self.time*100)
                texture = self.scene.ibl[id1].cuda()

            else:
                abc = self.trainable_abc()
                texture = self.trainable_abc.background_texture
                
            
            buffer_image = render(
                    cam,
                    self.gaussians,
                    abc,
                    texture,
                    view_args={
                        "vis_mode":self.vis_mode,
                        "stage":self.stage,
                        "finecoarse_flag":self.finecoarse_flag
                    },
            )

            try:
                buffer_image = buffer_image["render"]
            except:
                print(f'Mode "{self.vis_mode}" does not work')
                buffer_image = buffer_image['render']
                
            # if buffer_image.shape[0] == 1:
            #     buffer_image = (buffer_image - buffer_image.min())/(buffer_image.max() - buffer_image.min())
            #     buffer_image = buffer_image.repeat(3,1,1)


            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H,self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            
            try:
                if self.show_mask == 'occ':
                    mask = cam.sceneoccluded_mask.squeeze(0).cuda()
                else:
                    mask = 0.
                buffer_image[0] += mask*0.5
            except:
                pass
            
            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )



        t1 = time.time()
        
        buffer_image = self.buffer_image

        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!
        
        # Add _log_view_camera
        dpg.set_value("_log_view_camera", f"View {self.current_cam_index}")
        if 1./(t1-t0) < 500:
            dpg.set_value("_log_infer_time", f"{1./(t1-t0)} ")

    def save_scene(self):
        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
        self.scene.save(self.iteration, self.stage)
        print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
        torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_" + str(self.iteration) + ".pth")

    def register_dpg(self):
        
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")
            
        

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=400,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("N/A", tag="_log_infer_time")
            with dpg.group(horizontal=True):
                dpg.add_text("Stage: ")
                dpg.add_text("N/A", tag="_log_stage")
            # ----------------
            #  Loss Functions
            # ----------------
            with dpg.group():
                if self.view_test is False:
                    dpg.add_text("Training info:")
                    dpg.add_text("N/A", tag="_log_iter")
                    dpg.add_text("N/A", tag="_log_relit")
                    dpg.add_text("N/A", tag="_log_canon")
                    dpg.add_text("N/A", tag="_log_deform")
                    dpg.add_text("N/A", tag="_log_plane")
                    dpg.add_text("N/A", tag="_log_fine2")

                    dpg.add_text("N/A", tag="_log_points")
                else:
                    dpg.add_text("Training info: (Not training)")


            with dpg.collapsing_header(label="Testing info:", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Progress : ")
                    dpg.add_text("N/A", tag="_log_test_progress")

                dpg.add_text("Lighting-only : ")
                dpg.add_text("N/A", tag="_log_l_1")
                dpg.add_text("N/A", tag="_log_l_2")
                dpg.add_text("N/A", tag="_log_l_3")
                dpg.add_text("View-only : ")
                dpg.add_text("N/A", tag="_log_v_1")
                dpg.add_text("N/A", tag="_log_v_2")
                dpg.add_text("N/A", tag="_log_v_3")
                dpg.add_text("View & Lighting : ")
                dpg.add_text("N/A", tag="_log_lv_1")
                dpg.add_text("N/A", tag="_log_lv_2")
                dpg.add_text("N/A", tag="_log_lv_3")
                    

            # ----------------
            #  Control Functions
            # ----------------
            with dpg.collapsing_header(label="Rendering", default_open=True):
                def callback_toggle_show_rgb(sender):
                    self.switch_off_viewer = ~self.switch_off_viewer
                def callback_toggle_finecoarse(sender):
                    self.finecoarse_flag = False if self.finecoarse_flag else True
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Pause/Play Viewer ")
                    dpg.add_button(label="||>", callback=callback_toggle_show_rgb)  
                
                with dpg.group(horizontal=True):
                    dpg.add_text("View Relighting Scene")
                    dpg.add_button(label="--", callback=callback_toggle_finecoarse)
                     
                def callback_toggle_reset_cam(sender):
                    self.current_cam_index = 0
                    
                def callback_toggle_next_cam(sender):
                    self.current_cam_index = (self.current_cam_index + 1) % len(self.free_cams)
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset cam", callback=callback_toggle_reset_cam)
                    dpg.add_text("N/A", tag="_log_view_camera")
                    dpg.add_button(label="Next cam", callback=callback_toggle_next_cam)

                def callback_toggle_reset_cam(sender):
                    for i in range(len(self.free_cams)):
                        self.free_cams[i] = copy.deepcopy(self.original_cams[i])
                    self.current_cam_index = 0
            
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset Fov", callback=callback_toggle_reset_cam)
                    
                def callback_toggle_sceneocc_mask(sender):
                    self.show_mask = 'occ'
                def callback_toggle_no_mask(sender):
                    self.show_mask = 'none'
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Show/Unshow Masks")
                    dpg.add_button(label="No Mask", callback=callback_toggle_no_mask)
                    dpg.add_button(label="Occlu Mask", callback=callback_toggle_sceneocc_mask)
                    
                
                def callback_toggle_show_rgb(sender):
                    self.vis_mode = 'render'
                def callback_toggle_show_depth(sender):
                    self.vis_mode = 'D'
                def callback_toggle_show_edepth(sender):
                    self.vis_mode = 'ED'
                def callback_toggle_show_2dgsdepth(sender):
                    self.vis_mode = '2D'
                def callback_toggle_show_norms(sender):
                    self.vis_mode = 'normals'
                def callback_toggle_show_alpha(sender):
                    self.vis_mode = 'alpha'
                def callback_toggle_show_invariance(sender):
                    self.vis_mode = 'invariance'
                def callback_toggle_show_deform(sender):
                    self.vis_mode = 'deform'

                with dpg.group(horizontal=True):
                    dpg.add_button(label="RGB", callback=callback_toggle_show_rgb)
                    # dpg.add_button(label="Norms", callback=callback_toggle_show_norms)
                    # dpg.add_button(label="Alpha", callback=callback_toggle_show_alpha)
                    dpg.add_button(label="Invar", callback=callback_toggle_show_invariance)
                    dpg.add_button(label="Deform", callback=callback_toggle_show_deform)
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="D", callback=callback_toggle_show_depth)
                    dpg.add_button(label="ED", callback=callback_toggle_show_edepth)
                    dpg.add_button(label="2D", callback=callback_toggle_show_2dgsdepth)
                
                def callback_speed_control(sender):
                    self.time = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="Time",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )
                
        
        def zoom_callback_fov(sender, app_data):
            delta = app_data  # scroll: +1 = up (zoom in), -1 = down (zoom out)
            cam = self.free_cams[self.current_cam_index]

            zoom_scale = 0.5  # Smaller = faster zoom

            # Scale FoV within limits
            cam.fx *= zoom_scale if delta > 0 else 1 / zoom_scale
            cam.fy *= zoom_scale if delta > 0 else 1 / zoom_scale

        

        def drag_callback(sender, app_data):
            # app_data = (button, rel_x, rel_y)
            # button: 0=left, 1=right, 2=middle
            button, rel_x, rel_y = app_data
            
            if button != 0:  # only left drag
                return

            # simply check inside primary window dimensions
            if dpg.get_active_window() != dpg.get_alias_id("_primary_window"):
                return
            
            cam = self.free_cams[self.current_cam_index]

            # Sensitivity
            yaw_speed = -0.0001
            pitch_speed = 0.0001

            # Convert mouse drag into yaw/pitch rotations
            yaw = rel_x * yaw_speed
            pitch = rel_y * pitch_speed

            # --- Rotation Matrices ---
            Ry = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])

            R_drag = Rx @ Ry

            R_c2w = cam.R

            R_c2w_new = R_drag @ R_c2w
            
            cam.R = R_c2w_new
                
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=zoom_callback_fov)
            dpg.add_mouse_drag_handler(callback=drag_callback)
            
            
        dpg.create_viewport(
            title=f"{self.runname}",
            width=self.W + 400,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        
        
            
        dpg.setup_dearpygui()

        dpg.show_viewport()
        
from scipy.ndimage import distance_transform_edt
def get_in_view_dyn_mask(camera, xyz, X, Y) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (
        (ndc[:, 0].abs() <= 1) &
        (ndc[:, 1].abs() <= 1) &
        (ndc[:, 2].abs() <= 1)
    )
    visible_mask = in_ndc_bounds & in_front

    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
    py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
    mask = (1.-camera.scene_mask).to(device).squeeze(0)

    sampled_mask = mask[py_valid, px_valid].bool()

    H, W = camera.image_height, camera.image_width
    xyz_img = torch.zeros((H, W, 3), device=device)

    # Assign xyz into pixels
    px_final = px_valid[sampled_mask]
    py_final = py_valid[sampled_mask]
    xyz_vals = xyz[valid_idx][sampled_mask]
    xyz_img[py_final, px_final] = xyz_vals
    
    
    show = False
    if show:
        import matplotlib.pyplot as plt
        # Convert to CPU numpy for visualization
        img_np = xyz_img.detach().cpu().numpy()

        # Normalize for display (independently per channel)
        img_min = img_np.min(axis=(0, 1), keepdims=True)
        img_max = img_np.max(axis=(0, 1), keepdims=True)
        img_norm = (img_np - img_min) / (img_max - img_min + 1e-8)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Show normalized xyz image
        ax[0].imshow(img_norm)
        ax[0].set_title("Normalized xyz image")
        ax[0].axis("off")

        # Show XY indexing (scatter)
        ax[1].imshow(img_norm)
        ax[1].scatter(X, Y, s=3, c="red")
        ax[1].set_title("With XY indexing")
        ax[1].axis("off")

        plt.show()
        return None
    
    # --- Nearest-neighbor fill for empty pixels ---
    xyz_np = xyz_img.cpu().numpy()   # [H, W, 3]
    valid_mask = (xyz_np.sum(axis=-1) != 0)

    # distance_transform_edt returns for each empty pixel the index of the nearest valid pixel
    dist, indices = distance_transform_edt(~valid_mask,
                                           return_indices=True)
    filled = xyz_np[indices[0], indices[1]]  # nearest xyz per pixel

    point = filled[Y, X, :]
    
    show = False
    if show:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter all visible point cloud
        ax.scatter(
            xyz_vals[:, 0].cpu(),
            xyz_vals[:, 1].cpu(),
            xyz_vals[:, 2].cpu(),
            s=1, c="blue", alpha=0.5, label="Point cloud"
        )

        # Scatter your selected points
        ax.scatter(
            point[ 0],
            point[ 1],
            point[ 2],
            s=60, c="red", marker="o", label="Filtered XYZ"
        )

        ax.set_title("3D Point Cloud with Filtered Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        plt.show()

    return torch.from_numpy(point).float().to(device)


def remove_screen_points(camera, xyz):
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (
        (ndc[:, 0].abs() <= 1) &
        (ndc[:, 1].abs() <= 1) &
        (ndc[:, 2].abs() <= 1)
    )
    visible_mask = in_front & in_ndc_bounds

    # Pixel coordinates for all points (will clamp to bounds)
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long().clamp(0, camera.image_width - 1)
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long().clamp(0, camera.image_height - 1)

    # Scene mask (1 = free, 0 = masked/occluded)
    mask_img = (camera.sceneoccluded_mask).to(device).squeeze(0)

    # Start with all points marked False (not removed)
    remove_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Only check points that are visible
    sampled_mask = mask_img[py[visible_mask], px[visible_mask]].bool()

    # Mark visible points inside the mask for removal
    remove_mask[visible_mask] = sampled_mask

    return remove_mask
    

