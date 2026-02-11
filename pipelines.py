import os
import sys
import math
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib

# Define project root for checkpoint paths
project_root = os.path.dirname(os.path.abspath(__file__))
    
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import FluxControlPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video

# from submodules.MoGe.moge.model.v1 import MoGeModel
from submodules.MoGe.moge.model.v2 import MoGeModel
from submodules.DELTA.densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from submodules.DELTA.densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from pi3.utils.basic import load_images_as_tensor
from pi3.models.pi3 import Pi3
from pi3.utils.geometry import se3_inverse

from image_gen_aux import DepthPreprocessor
from moviepy.editor import ImageSequenceClip
from einops import rearrange
from typing import List, Tuple
from packaging import version as pver

def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

class FirstFrameRepainter:
    def __init__(self, height = 480, width = 720 , gpu_id=0, output_dir='outputs'):
        """Initialize FirstFrameRepainter
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        self.device = f"cuda:{gpu_id}"
        self.output_dir = output_dir
        self.max_depth = 65.0
        self.height = height
        self.width = width
        os.makedirs(output_dir, exist_ok=True)
        
    def repaint(self, image_tensor, prompt, depth_path=None, method="dav"):
        """Repaint first frame using Flux
        
        Args:
            image_tensor (torch.Tensor): Input image tensor [C,H,W]
            prompt (str): Repaint prompt
            depth_path (str): Path to depth image
            method (str): depth estimator, "moge" or "dav" or "zoedepth"
            
        Returns:
            torch.Tensor: Repainted image tensor [C,H,W]
        """
        print("Loading Flux model...")
        # Load Flux model
        flux_pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev", 
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Get depth map
        if depth_path is None:
            if method == "moge":
                self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
                depth_map = self.moge_model.infer(image_tensor.to(self.device))["depth"]
                depth_map = torch.clamp(depth_map, max=self.max_depth)
                depth_normalized = 1.0 - (depth_map / self.max_depth)
                depth_rgb = (depth_normalized * 255).cpu().numpy().astype(np.uint8)
                control_image = Image.fromarray(depth_rgb).convert("RGB")
            elif method == "zoedepth":
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
                self.depth_preprocessor.to(self.device)
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(image_np))[0].convert("RGB")
                control_image = control_image.point(lambda x: 255 - x) # the zoedepth depth is inverted
            else:
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                self.depth_preprocessor.to(self.device)
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(image_np))[0].convert("RGB")
        else:
            control_image = Image.open(depth_path).convert("RGB")

        try:
            repainted_image = flux_pipe(
                prompt=prompt,
                control_image=control_image,
                height=self.height,
                width=self.width,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # Save repainted image
            repainted_image.save(os.path.join(self.output_dir, "temp_repainted.png"))
            
            # Convert PIL Image to tensor
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            repainted_tensor = transform(repainted_image)
            
            return repainted_tensor
            
        finally:
            # Clean up GPU memory
            del flux_pipe
            if method == "moge":
                del self.moge_model
            else:
                del self.depth_preprocessor
            torch.cuda.empty_cache()

class CameraMotionGenerator:
    def __init__(self, motion_type, frame_num=49, H=480, W=720, fx=None, fy=None, fov=55, device='cuda', pose_file=None):
        self.motion_type = motion_type
        self.frame_num = frame_num
        self.fov = fov
        self.device = device
        self.W = W
        self.H = H
        self.pose_file = pose_file
        self.intr = torch.tensor([
            [0, 0, W / 2],
            [0, 0, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        # if fx, fy not provided
        if not fx or not fy:
            fov_rad = math.radians(fov)
            fx = fy = (W / 2) / math.tan(fov_rad / 2)
 
        self.intr[0, 0] = fx
        self.intr[1, 1] = fy   

        self.extr = torch.eye(4, device=device)

    def process_pose_file(self, pose_file_path, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
        """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
        """
        with open(pose_file_path, 'r') as f:
            poses = f.readlines()

        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        if return_poses:
            cam_params = [Camera(cam_param) for cam_param in cam_params]
            return cam_params
        else:
            cam_params = [Camera(cam_param) for cam_param in cam_params]

            sample_wh_ratio = width / height
            pose_wh_ratio = original_pose_width / original_pose_height

            if pose_wh_ratio > sample_wh_ratio:
                resized_ori_w = height * pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / width
            else:
                resized_ori_h = width / pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / height

            intrinsic = np.asarray([[cam_param.fx * width,
                                    cam_param.fy * height,
                                    cam_param.cx * width,
                                    cam_param.cy * height]
                                    for cam_param in cam_params], dtype=np.float32)

            # Convert to 3x3 intrinsic matrix format for set_intr function (normalized)
            K = np.array([[intrinsic[0, 0]/width, 0, intrinsic[0, 2]/width],
                         [0, intrinsic[0, 1]/height, intrinsic[0, 3]/height],
                         [0, 0, 1]], dtype=np.float32)
            
            self.set_intr(K)

            return cam_params

    def process_video_file(self, video_file_path, width=672, height=384, device='cpu'):
        """Process video file to extract camera parameters using Pi3 model
        
        Args:
            video_file_path (str): Path to the video file
            width (int): Target width
            height (int): Target height  
            device (str): Device to use for processing
            
        Returns:
            List[Camera]: List of camera parameters for each frame
        """
        try:
            # Pi3 dependencies are imported at the top of the script
            
            # Load Pi3 model
            print(f"Loading Pi3 model...")
            pi3_device = torch.device(device)
            model = Pi3.from_pretrained("yyfz233/Pi3").to(pi3_device).eval()
            
            # Load video with interval=1
            print(f"Loading video: {video_file_path}")
            imgs = load_images_as_tensor(video_file_path, interval=1).to(pi3_device)  # (N, 3, H, W)
            
            # Run inference
            print("Running Pi3 model inference...")
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=dtype):
                    pred = model(imgs[None])  # Add batch dimension
            
            # Extract camera poses (c2w matrices)
            poses_c2w_all = pred['camera_poses'].cpu()  # Shape: [B, N, 4, 4]
            poses_c2w_all = poses_c2w_all[0]  # Remove batch dimension: [N, 4, 4]

            # Convert c2w matrices to Camera objects
            cam_params = []
            for i, c2w_matrix in enumerate(poses_c2w_all):
                # Convert c2w to w2c
                c2w_np = c2w_matrix.numpy()
                w2c_np = se3_inverse(c2w_np)
                
                # Extract intrinsic parameters (assuming default values, can be refined)
                fx = (width / 2) / math.tan(math.radians(55) / 2)  # Default FOV of 55 degrees
                fy = fx
                cx = width / 2
                cy = height / 2
                
                # Create entry format expected by Camera class
                # Format: [frame_id, fx, fy, cx, cy, 0, 0, w2c_matrix_flattened]
                w2c_flat = w2c_np[:3, :].flatten()  # Convert 3x4 matrix to flat array
                entry = [i, fx, fy, cx, cy, 0, 0] + w2c_flat.tolist()
                
                cam_params.append(Camera(entry))
            
            print(f"Extracted camera poses for {len(cam_params)} frames")
            return cam_params
            
        except ImportError as e:
            raise ImportError(f"Failed to import Pi3 dependencies: {e}. Please ensure Pi3 is properly installed.")
        except Exception as e:
            raise RuntimeError(f"Error processing video file {video_file_path}: {e}")

    def convert_cameras_to_poses(self, intrinsic_list, extrinsic_list):
        """Convert loaded camera parameters to pose matrices
        
        Args:
            intrinsic_list (List[List[float]]): List of intrinsic parameters [fx, fy, cx, cy]
            extrinsic_list (List[List[List[float]]]): List of extrinsic matrices (3x4)
            
        Returns:
            torch.Tensor: Camera poses of shape [num_frames, 4, 4]
        """
        poses = []
        for intrinsic, extrinsic in zip(intrinsic_list, extrinsic_list):
            # Convert extrinsic from list to numpy array
            extr_matrix = np.array(extrinsic)  # 3x4 matrix
            
            # Create 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :4] = extr_matrix
            poses.append(pose)
        
        # Convert to tensor and ensure we have the right number of frames
        poses_array = np.array(poses)
        
        # If we have fewer poses than frames, repeat the last pose
        if len(poses) < self.frame_num:
            last_pose = poses_array[-1:].repeat(self.frame_num - len(poses), axis=0)
            poses_array = np.concatenate([poses_array, last_pose], axis=0)
        # If we have more poses than frames, truncate
        elif len(poses) > self.frame_num:
            poses_array = poses_array[:self.frame_num]
        
        return torch.from_numpy(poses_array).to(self.device).float()

    def s2w_vggt(self, points, extrinsics, intrinsics):
        """
        Transform points from pixel coordinates to world coordinates
        
        Args:
            points: Point cloud data of shape [T, N, 3] in uvz format
            extrinsics: Camera extrinsic matrices [B, T, 3, 4] or [T, 3, 4]
            intrinsics: Camera intrinsic matrices [B, T, 3, 3] or [T, 3, 3]
            
        Returns:
            world_points: Point cloud in world coordinates [T, N, 3]
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.detach().cpu().numpy()
            # Handle batch dimension
            if extrinsics.ndim == 4:  # [B, T, 3, 4]
                extrinsics = extrinsics[0]  # Take first batch
            
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
            # Handle batch dimension
            if intrinsics.ndim == 4:  # [B, T, 3, 3]
                intrinsics = intrinsics[0]  # Take first batch
            
        T, N, _ = points.shape
        world_points = np.zeros_like(points)
        
        # Extract uvz coordinates
        uvz = points
        valid_mask = uvz[..., 2] > 0
        
        # Create homogeneous coordinates [u, v, 1]
        uv_homogeneous = np.concatenate([uvz[..., :2], np.ones((T, N, 1))], axis=-1)
        
        # Transform from pixel to camera coordinates
        for i in range(T):
            K = intrinsics[i]
            K_inv = np.linalg.inv(K)
            
            R = extrinsics[i, :, :3]
            t = extrinsics[i, :, 3]
            
            R_inv = np.linalg.inv(R)
            
            valid_indices = np.where(valid_mask[i])[0]
            
            if len(valid_indices) > 0:
                valid_uv = uv_homogeneous[i, valid_indices]
                valid_z = uvz[i, valid_indices, 2]
                
                valid_xyz_camera = valid_uv @ K_inv.T
                valid_xyz_camera = valid_xyz_camera * valid_z[:, np.newaxis]
                
                # Transform from camera to world coordinates: X_world = R^-1 * (X_camera - t)
                valid_world_points = (valid_xyz_camera - t) @ R_inv.T
                
                world_points[i, valid_indices] = valid_world_points
        
        return world_points

    def w2s_vggt(self, world_points, extrinsics, intrinsics, poses=None, override_extrinsics=True):
        """
        Project points from world coordinates to camera view
        
        Args:
            world_points: Point cloud in world coordinates [T, N, 3]
            extrinsics: Original camera extrinsic matrices [B, T, 3, 4] or [T, 3, 4]
            intrinsics: Camera intrinsic matrices [B, T, 3, 3] or [T, 3, 3]
            poses: Camera pose matrices [T, 4, 4], if None use first frame extrinsics
            override_extrinsics: If True, replace extrinsics with poses; if False, apply poses on top of extrinsics
            
        Returns:
            camera_points: Point cloud in camera coordinates [T, N, 3] in uvz format
        """
        if isinstance(world_points, torch.Tensor):
            world_points = world_points.detach().cpu().numpy()
            
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.detach().cpu().numpy()
            if extrinsics.ndim == 4:
                extrinsics = extrinsics[0]
            
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
            if intrinsics.ndim == 4:
                intrinsics = intrinsics[0]
            
        T, N, _ = world_points.shape
        
        # If no poses provided, use first frame extrinsics
        if poses is None:
            pose1 = np.eye(4)
            pose1[:3, :3] = extrinsics[0, :, :3]
            pose1[:3, 3] = extrinsics[0, :, 3]
            
            camera_poses = np.tile(pose1[np.newaxis, :, :], (T, 1, 1))
        else:
            if isinstance(poses, torch.Tensor):
                camera_poses = poses.cpu().numpy()
            else:
                camera_poses = poses
            
            # Scale translation by 1/5
            scaled_poses = camera_poses.copy()
            scaled_poses[:, :3, 3] = camera_poses[:, :3, 3] / 5.0
            camera_poses = scaled_poses
            
            # If not overriding extrinsics, combine poses with original extrinsics
            if not override_extrinsics and poses is not None:
                for i in range(T):
                    # Convert extrinsics to 4x4 matrix
                    ext_mat = np.eye(4)
                    ext_mat[:3, :3] = extrinsics[i, :, :3]
                    ext_mat[:3, 3] = extrinsics[i, :, 3]
                    
                    # Combine pose with extrinsics: pose * extrinsics
                    combined = np.matmul(camera_poses[i], ext_mat)
                    
                    # Update camera_poses
                    camera_poses[i] = combined
        
        # Add homogeneous coordinates
        ones = np.ones([T, N, 1])
        world_points_hom = np.concatenate([world_points, ones], axis=-1)
        
        # Transform points using batch matrix multiplication
        pts_cam_hom = np.matmul(world_points_hom, np.transpose(camera_poses, (0, 2, 1)))
        pts_cam = pts_cam_hom[..., :3]
        
        # Extract depth information
        depths = pts_cam[..., 2:3]
        valid_mask = depths[..., 0] > 0
        
        # Normalize coordinates
        normalized_pts = pts_cam / (depths + 1e-10)
        
        # Apply intrinsic matrix for projection
        pts_pixel = np.matmul(normalized_pts, np.transpose(intrinsics, (0, 2, 1)))
        
        # Extract pixel coordinates
        u = pts_pixel[..., 0:1]
        v = pts_pixel[..., 1:2]
        
        # Set invalid points to zero
        u[~valid_mask] = 0
        v[~valid_mask] = 0
        depths[~valid_mask] = 0
        
        # Return points in uvz format
        result = np.concatenate([u, v, depths], axis=-1)
        
        return torch.from_numpy(result)
    
    def w2s_moge(self, pts, poses):
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        assert poses.shape[0] == self.frame_num
        poses = poses.to(torch.float32).to(self.device)
        T, N, _ = pts.shape  # (T, N, 3)
        intr = self.intr.unsqueeze(0).repeat(self.frame_num, 1, 1)
        ones = torch.ones((T, N, 1), device=self.device, dtype=pts.dtype)
        points_world_h = torch.cat([pts, ones], dim=-1)
        points_camera_h = torch.bmm(poses, points_world_h.permute(0, 2, 1))
        points_camera = points_camera_h[:, :3, :].permute(0, 2, 1)

        points_image_h = torch.bmm(points_camera, intr.permute(0, 2, 1))

        uv = points_image_h[:, :, :2] / points_image_h[:, :, 2:3]
        depth = points_camera[:, :, 2:3]  # (T, N, 1)
        uvd = torch.cat([uv, depth], dim=-1)  # (T, N, 3)

        return uvd
    
    def set_intr(self, K):
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.intr = K.to(self.device)
        print(self.intr)

    def set_extr(self, extr):
        if isinstance(extr, np.ndarray):    
            extr = torch.from_numpy(extr)
        self.extr = extr.to(self.device)

    def rot_poses(self, angle, axis='y'):
        """Generate a single rotation matrix
        
        Args:
            angle (float): Rotation angle in degrees
            axis (str): Rotation axis ('x', 'y', or 'z')
            
        Returns:
            torch.Tensor: Single rotation matrix [4, 4]
        """
        angle_rad = math.radians(angle)
        cos_theta = torch.cos(torch.tensor(angle_rad))
        sin_theta = torch.sin(torch.tensor(angle_rad))
        
        if axis == 'x':
            rot_mat = torch.tensor([
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        elif axis == 'y':
            rot_mat = torch.tensor([
                [cos_theta, 0, sin_theta, 0],
                [0, 1, 0, 0],
                [-sin_theta, 0, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        elif axis == 'z':
            rot_mat = torch.tensor([
                [cos_theta, -sin_theta, 0, 0],
                [sin_theta, cos_theta, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        else:
            raise ValueError("Invalid axis value. Choose 'x', 'y', or 'z'.")
            
        return rot_mat.to(self.device)

    def trans_poses(self, dx, dy, dz):
        """
        params:
        - dx: float, displacement along x axis。
        - dy: float, displacement along y axis。
        - dz: float, displacement along z axis。

        ret:
        - matrices: torch.Tensor
        """
        trans_mats = torch.eye(4).unsqueeze(0).repeat(self.frame_num, 1, 1)  # (n, 4, 4)

        delta_x = dx / (self.frame_num - 1)
        delta_y = dy / (self.frame_num - 1)
        delta_z = dz / (self.frame_num - 1)

        for i in range(self.frame_num):
            trans_mats[i, 0, 3] = i * delta_x
            trans_mats[i, 1, 3] = i * delta_y
            trans_mats[i, 2, 3] = i * delta_z

        return trans_mats.to(self.device)
    

    def _look_at(self, camera_position, target_position):
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        # calculate rotation matrix
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction])
        rotation_matrix = np.linalg.inv(rotation_matrix)
        return rotation_matrix

    def spiral_poses(self, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
        """Generate spiral camera poses
        
        Args:
            radius (float): Base radius of the spiral
            forward_ratio (float): Scale factor for forward motion
            backward_ratio (float): Scale factor for backward motion
            rotation_times (float): Number of rotations to complete
            look_at_times (float): Scale factor for look-at point distance
            
        Returns:
            torch.Tensor: Camera poses of shape [num_frames, 4, 4]
        """
        # Generate spiral trajectory
        t = np.linspace(0, 1, self.frame_num)
        r = np.sin(np.pi * t) * radius * rotation_times
        theta = 2 * np.pi * t
        
        # Calculate camera positions
        # Limit y motion for better floor/sky view
        y = r * np.cos(theta) * 0.15
        x = r * np.sin(theta) * 0.5 
        z = -r
        z[z < 0] *= forward_ratio
        z[z > 0] *= backward_ratio
        
        # Set look-at target
        target_pos = np.array([0, 0, radius * look_at_times])
        cam_pos = np.vstack([x, y, z]).T
        cam_poses = []
        
        for pos in cam_pos:
            rot_mat = self._look_at(pos, target_pos)
            trans_mat = np.eye(4)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = pos
            cam_poses.append(trans_mat[None])
            
        camera_poses = np.concatenate(cam_poses, axis=0)
        return torch.from_numpy(camera_poses).to(self.device)

    def get_default_motion(self):
        """Parse motion parameters and generate corresponding motion matrices
        
        Supported formats:
        - trans <dx> <dy> <dz> [start_frame] [end_frame]: Translation motion
        - rot <axis> <angle> [start_frame] [end_frame]: Rotation motion
        - spiral <radius> [start_frame] [end_frame]: Spiral motion
        - path: Load camera poses from text file specified in camera_txt parameter
        
        Multiple transformations can be combined using semicolon (;) as separator:
        e.g., "trans 0 0 0.5 0 30; rot x 25 0 30; trans 0.1 0 0 30 48"
        
        Note:
            - start_frame and end_frame are optional
            - frame range: 0-(frame_num-1) (will be clamped to this range)
            - if not specified, defaults to 0-(frame_num-1)
            - frames after end_frame will maintain the final transformation
            - for combined transformations, they are applied in sequence
            - moving left, up and zoom out is positive in video
        
        Returns:
            torch.Tensor: Motion matrices [num_frames, 4, 4]
        """
        # se3_inverse is imported at the top of the script
        
        if not isinstance(self.motion_type, str):
            raise ValueError(f'camera_motion must be a string, but got {type(self.motion_type)}')
        
        # Split combined transformations
        transform_sequences = [s.strip() for s in self.motion_type.split(';')]
        
        # Initialize the final motion matrices
        final_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.frame_num, 1, 1)
        
        # Process each transformation in sequence
        for transform in transform_sequences:
            params = transform.lower().split()
            if not params:
                continue
                
            motion_type = params[0]
            
            # Default frame range
            start_frame = 0
            end_frame = self.frame_num - 1
            
            if motion_type == 'trans':
                # Parse translation parameters
                if len(params) not in [4, 6]:
                    raise ValueError(f"trans motion requires 3 or 5 parameters: 'trans <dx> <dy> <dz>' or 'trans <dx> <dy> <dz> <start_frame> <end_frame>', got: {transform}")
                
                dx, dy, dz = map(float, params[1:4])
                
                if len(params) == 6:
                    start_frame = max(0, min(self.frame_num - 1, int(params[4])))
                    end_frame = max(0, min(self.frame_num - 1, int(params[5])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                # Generate current transformation
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.frame_num, 1, 1)
                for frame_idx in range(self.frame_num):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        current_motion[frame_idx, :3, 3] = torch.tensor([dx, dy, dz], device=self.device) * t
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'rot':
                # Parse rotation parameters
                if len(params) not in [3, 5]:
                    raise ValueError(f"rot motion requires 2 or 4 parameters: 'rot <axis> <angle>' or 'rot <axis> <angle> <start_frame> <end_frame>', got: {transform}")
                
                axis = params[1]
                if axis not in ['x', 'y', 'z']:
                    raise ValueError(f"Invalid rotation axis '{axis}', must be 'x', 'y' or 'z'")
                angle = float(params[2])
                
                if len(params) == 5:
                    start_frame = max(0, min(self.frame_num - 1, int(params[3])))
                    end_frame = max(0, min(self.frame_num - 1, int(params[4])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.frame_num, 1, 1)
                for frame_idx in range(self.frame_num):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        current_angle = angle * t
                        current_motion[frame_idx] = self.rot_poses(current_angle, axis)
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'spiral':
                # Parse spiral motion parameters
                if len(params) not in [2, 4]:
                    raise ValueError(f"spiral motion requires 1 or 3 parameters: 'spiral <radius>' or 'spiral <radius> <start_frame> <end_frame>', got: {transform}")
                
                radius = float(params[1])
                
                if len(params) == 4:
                    start_frame = max(0, min(self.frame_num - 1, int(params[2])))
                    end_frame = max(0, min(self.frame_num - 1, int(params[3])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.frame_num, 1, 1)
                spiral_motion = self.spiral_poses(radius)
                for frame_idx in range(self.frame_num):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        idx = int(t * (len(spiral_motion) - 1))
                        current_motion[frame_idx] = spiral_motion[idx]
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'path':
                # Load camera poses from file
                if self.pose_file is None:
                    raise ValueError("pose_file must be provided when using 'path' motion type")
                
                # Check file extension to determine processing method
                file_ext = os.path.splitext(self.pose_file)[1].lower()
                
                if file_ext == '.txt':
                    # Load camera parameters from text file using process_pose_file
                    # Get camera parameters in raw format 
                    cam_params = self.process_pose_file(
                        self.pose_file, 
                        width=self.W, 
                        height=self.H,
                        device=self.device,
                        return_poses=True
                    )
                elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    # Process video input using Pi3 to estimate camera parameters
                    cam_params = self.process_video_file(
                        self.pose_file,
                        width=self.W,
                        height=self.H,
                        device=self.device
                    )
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .txt, .mp4, .avi, .mov, .mkv, .webm")
                
                # Use get_relative_pose to get relative camera poses
                relative_c2ws = get_relative_pose(cam_params)  # Returns relative c2w matrices
                
                # Convert c2w to w2c matrices (what get_default_motion expects)
                w2c_poses = []
                for c2w in relative_c2ws:
                    w2c = se3_inverse(c2w)  # Convert c2w to w2c
                    w2c_poses.append(w2c)

                # Convert to tensor and ensure we have the right number of frames
                poses_array = np.array(w2c_poses)
                
                # If we have fewer poses than frames, repeat the last pose
                if len(poses_array) < self.frame_num:
                    last_pose = poses_array[-1:].repeat(self.frame_num - len(poses_array), axis=0)
                    poses_array = np.concatenate([poses_array, last_pose], axis=0)
                # If we have more poses than frames, truncate
                elif len(poses_array) > self.frame_num:
                    poses_array = poses_array[:self.frame_num]
                
                loaded_poses = torch.from_numpy(poses_array).to(self.device).float()
                
                # Use loaded poses directly as the final motion
                final_motion = loaded_poses
                break  # No need to process more transforms when using path
                
            else:
                raise ValueError(f'camera_motion type must be in [trans, spiral, rot, path], but got {motion_type}')
        
        return final_motion

class ObjectMotionGenerator:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.num_frames = 49
        
    def _get_points_in_mask(self, pred_tracks, mask):
        """Get points that lie within the mask
        
        Args:
            pred_tracks (torch.Tensor): Point trajectories [num_frames, num_points, 3] 
            mask (torch.Tensor): Binary mask [H, W]
            
        Returns:
            torch.Tensor: Boolean mask for selected points [num_points]
        """
        first_frame_points = pred_tracks[0]  # [num_points, 3]
        xy_points = first_frame_points[:, :2]  # [num_points, 2]
        
        xy_pixels = xy_points.round().long()
        xy_pixels[:, 0].clamp_(0, mask.shape[1] - 1)
        xy_pixels[:, 1].clamp_(0, mask.shape[0] - 1)
        
        points_in_mask = mask[xy_pixels[:, 1], xy_pixels[:, 0]]
        
        return points_in_mask

    def apply_motion(self, pred_tracks, mask, motion_type, distance, num_frames=49, tracking_method="DELTA"):

        self.num_frames = num_frames
        pred_tracks = pred_tracks.to(self.device).float()
        mask = mask.to(self.device)

        template = {
            'up': ('trans', torch.tensor([0, -1, 0])),
            'down': ('trans', torch.tensor([0, 1, 0])), 
            'left': ('trans', torch.tensor([-1, 0, 0])),
            'right': ('trans', torch.tensor([1, 0, 0])),
            'front': ('trans', torch.tensor([0, 0, 1])),
            'back': ('trans', torch.tensor([0, 0, -1])),

            # 对角线移动 (2D平面)
            'up_left': ('trans', torch.tensor([-1, -1, 0]) / math.sqrt(2)),
            'up_right': ('trans', torch.tensor([1, -1, 0]) / math.sqrt(2)),
            'down_left': ('trans', torch.tensor([-1, 1, 0]) / math.sqrt(2)),
            'down_left2': ('trans', torch.tensor([-1, 0.5, 0]) / math.sqrt(2)),
            'down_right': ('trans', torch.tensor([1, 1, 0]) / math.sqrt(2)),
            
            # 3D空间对角移动
            'up_front': ('trans', torch.tensor([0, -1, 1]) / math.sqrt(2)),
            'up_back': ('trans', torch.tensor([0, -1, -1]) / math.sqrt(2)),
            'down_front': ('trans', torch.tensor([0, 1, 1]) / math.sqrt(2)),
            'down_back': ('trans', torch.tensor([0, 1, -1]) / math.sqrt(2)),
            'left_front': ('trans', torch.tensor([-1, 0, 1]) / math.sqrt(2)),
            'left_back': ('trans', torch.tensor([-1, 0, -1]) / math.sqrt(2)),
            'right_front': ('trans', torch.tensor([1, 0, 1]) / math.sqrt(2)),
            'right_back': ('trans', torch.tensor([1, 0, -1]) / math.sqrt(2)),
            
            # 3维空间对角移动 (8个角)
            'up_left_front': ('trans', torch.tensor([-1, -1, 1]) / math.sqrt(3)),
            'up_left_back': ('trans', torch.tensor([-1, -1, -1]) / math.sqrt(3)),
            'up_right_front': ('trans', torch.tensor([1, -1, 1]) / math.sqrt(3)),
            'up_right_back': ('trans', torch.tensor([1, -1, -1]) / math.sqrt(3)),
            'down_left_front': ('trans', torch.tensor([-1, 1, 1]) / math.sqrt(3)),
            'down_left_back': ('trans', torch.tensor([-1, 1, -1]) / math.sqrt(3)),
            'down_right_front': ('trans', torch.tensor([1, 1, 1]) / math.sqrt(3)),
            'down_right_back': ('trans', torch.tensor([1, 1, -1]) / math.sqrt(3)),
            

            'rot': ('rot', None), # rotate around y axis (clockwise)
            'rot_ccw': ('rot_ccw', None), # rotate around y axis (counter-clockwise)
            
            # 头部旋转
            'pitch_up': ('rot_x', None), # 向上抬头 (rotation around x axis)
            'pitch_down': ('rot_x_ccw', None), # 向下低头 (rotation around x axis counter-clockwise)
            'roll_left': ('rot_z', None), # 向左侧头 (rotation around z axis)
            'roll_right': ('rot_z_ccw', None) # 向右侧头 (rotation around z axis counter-clockwise)
        }
        
        if motion_type not in template:
            raise ValueError(f"unknown motion type: {motion_type}")
            
        motion_type, base_vec = template[motion_type]
        if base_vec is not None:
            base_vec = base_vec.to(self.device) * distance

        if tracking_method == "moge":
            T, H, W, _ = pred_tracks.shape
            valid_selected = ~torch.any(torch.isnan(pred_tracks[0]), dim=2) & mask
            points = pred_tracks[0][valid_selected].reshape(-1, 3)
        else:
            points_in_mask = self._get_points_in_mask(pred_tracks, mask)
            points = pred_tracks[0, points_in_mask]
            
        center = points.mean(dim=0)
        
        motions = []
        for frame_idx in range(num_frames):
            t = frame_idx / (num_frames - 1)
            current_motion = torch.eye(4, device=self.device)
            current_motion[:3, 3] = -center
            motion_mat = torch.eye(4, device=self.device)
            if motion_type == 'trans':
                motion_mat[:3, 3] = base_vec * t
            elif motion_type == 'rot':
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[0, 0] = cos_t
                motion_mat[0, 2] = sin_t
                motion_mat[2, 0] = -sin_t
                motion_mat[2, 2] = cos_t
            elif motion_type == 'rot_ccw':
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[0, 0] = cos_t
                motion_mat[0, 2] = -sin_t
                motion_mat[2, 0] = sin_t
                motion_mat[2, 2] = cos_t
            elif motion_type == 'rot_x':  # pitch up
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[1, 1] = cos_t
                motion_mat[1, 2] = -sin_t
                motion_mat[2, 1] = sin_t
                motion_mat[2, 2] = cos_t
            elif motion_type == 'rot_x_ccw':  # pitch down
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[1, 1] = cos_t
                motion_mat[1, 2] = sin_t
                motion_mat[2, 1] = -sin_t
                motion_mat[2, 2] = cos_t
            elif motion_type == 'rot_z':  # roll left
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[0, 0] = cos_t
                motion_mat[0, 1] = -sin_t
                motion_mat[1, 0] = sin_t
                motion_mat[1, 1] = cos_t
            else:  # 'rot_z_ccw' - roll right
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[0, 0] = cos_t
                motion_mat[0, 1] = sin_t
                motion_mat[1, 0] = -sin_t
                motion_mat[1, 1] = cos_t
            
            current_motion = motion_mat @ current_motion
            current_motion[:3, 3] += center
            motions.append(current_motion)
            
        motions = torch.stack(motions)  # [num_frames, 4, 4]

        if tracking_method == "moge":
            modified_tracks = pred_tracks.clone().reshape(T, -1, 3)
            valid_selected = valid_selected.reshape([-1])

            for frame_idx in range(self.num_frames):
                motion_mat = motions[frame_idx]
                if W > 1: 
                    motion_mat = motion_mat.clone()
                    motion_mat[0, 3] /= W
                    motion_mat[1, 3] /= H
                points = modified_tracks[frame_idx, valid_selected]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, valid_selected] = transformed_points[:, :3]
            
            return modified_tracks.reshape(T, H, W, 3)
            
        else:
            points_in_mask = self._get_points_in_mask(pred_tracks, mask)
            modified_tracks = pred_tracks.clone()
            
            for frame_idx in range(pred_tracks.shape[0]):
                motion_mat = motions[frame_idx]
                points = modified_tracks[frame_idx, points_in_mask]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, points_in_mask] = transformed_points[:, :3]
            
            return modified_tracks

class FlexAMPipeline:
    def __init__(self, gpu_id=0, output_dir='outputs'):
        """Initialize MotionTransfer class
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        # video parameters
        self.max_depth = 81
        self.fps = 16

        # camera parameters
        self.camera_motion=None
        self.fov=55

        # device
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        self.dtype = torch.bfloat16

        # files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])

    @torch.no_grad()
    def _infer(
        self, 
        prompt: str,
        model_path: str,
        tracking_tensor: torch.Tensor = None,
        cos_video_dict: dict = None,
        depth_video : torch.Tensor = None,
        cos_level: int = 4, 
        full_ref: torch.Tensor = None,
        inpaint_video: torch.Tensor = None,
        inpaint_video_mask: torch.Tensor = None,
        output_path: str = "./output.mp4",
        num_inference_steps: int = 40,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        fps: int = 24,
        video_length: int = 81, 
        height: int = 480, 
        width: int = 720,
        seed: int = 42,
        density: int = 10,
    ):
        """
        Generates a video based on the given prompt and saves it to the specified path.

        Parameters:
        - prompt (str): The description of the video to be generated.
        - model_path (str): The path of the pre-trained model to be used.
        - tracking_tensor (torch.Tensor): Tracking video tensor [T, C, H, W] in range [0,1]
        - image_tensor (torch.Tensor): Input image tensor [C, H, W] in range [0,1]
        - output_path (str): The path where the generated video will be saved.
        - num_inference_steps (int): Number of steps for the inference process.
        - guidance_scale (float): The scale for classifier-free guidance.
        - num_videos_per_prompt (int): Number of videos to generate per prompt.
        - dtype (torch.dtype): The data type for computation.
        - seed (int): The seed for reproducibility.
        """
        from FlexAM.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel, WanT5EncoderModel, Wan2_2Transformer3DModel_FlexAM)
        from FlexAM.pipeline import Wan2_2FunControlPipeline_FlexAM
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from diffusers import FlowMatchEulerDiscreteScheduler
        from FlexAM.utils.utils import filter_kwargs
        from FlexAM.utils.utils import (save_videos_grid)
        config_path         = "config/wan2.2/wan_civitai_5b_FlexAM.yaml"
        config = OmegaConf.load(config_path)

        transformer = Wan2_2Transformer3DModel_FlexAM.from_pretrained(
            os.path.join(model_path, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            # low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

        # Get Vae
        vae = AutoencoderKLWan3_8.from_pretrained(
            os.path.join(model_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(dtype)
        # Get Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        text_encoder = text_encoder.eval()

        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        pipe = Wan2_2FunControlPipeline_FlexAM(
            transformer=transformer,
            transformer_2=None,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        ).to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.dtype = dtype
        # pipe.enable_sequential_cpu_offload()
        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        # 4. Generate the video frames based on the prompt.
        video_generate = pipe(
            prompt=prompt, 
            num_frames=video_length,
            negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, work, painting, picture, still, gray overall, worst quality, low quality, JPEG compression residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, morphomorphous limbs, finger fusion, still picture, messy background, three legs, a lot of people in the background, walking backwards",
            height=height,
            width=width,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=50,
            control_video = tracking_tensor,
            depth_video = depth_video,
            cos_level= cos_level,
            cos_control_videos = cos_video_dict,
            density=1/density,
            control_camera_video=None,
            ref_image=full_ref, # for full_ref
            video      = inpaint_video,
            mask_video   = inpaint_video_mask,
        ).videos

        # 5. Export the generated frames to a video file. fps must be 8 for original video.
        output_path = output_path if output_path else f"result.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_videos_grid(video_generate, output_path, fps=fps)

        
    #========== camera parameters ==========#

    def _set_camera_motion(self, camera_motion):
        self.camera_motion = camera_motion
    
    ##============= MoGe =============##

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds
        
        Args:
            pixels (numpy.ndarray): Pixel coordinates of shape [N, 2]
            W (int): Image width
            H (int): Image height
            
        Returns:
            numpy.ndarray: Boolean mask of valid pixels
        """
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values
        
        Args:
            points (numpy.ndarray): Points array of shape [N, 2]
            depths (numpy.ndarray): Depth values of shape [N]
            
        Returns:
            tuple: (sorted_points, sorted_depths, sort_index)
        """
        # Combine points and depths into a single array for sorting
        combined = np.hstack((points, depths[:, None]))  # Nx3 (points + depth)
        # Sort by depth (last column) in descending order
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
        # Split back into points and depths
        sorted_points = sorted_combined[:, :-1]
        sorted_depths = sorted_combined[:, -1]
        return sorted_points, sorted_depths, sort_index

    def draw_rectangle(self, rgb, coord, side_length, color=(255, 0, 0)):
        """Draw a rectangle on the image
        
        Args:
            rgb (PIL.Image): Image to draw on
            coord (tuple): Center coordinates (x, y)
            side_length (int): Length of rectangle sides
            color (tuple): RGB color tuple
        """
        draw = ImageDraw.Draw(rgb)
        # Calculate the bounding box of the rectangle
        left_up_point = (coord[0] - side_length//2, coord[1] - side_length//2)  
        right_down_point = (coord[0] + side_length//2, coord[1] + side_length//2)
        color = tuple(list(color))

        draw.rectangle(
            [left_up_point, right_down_point],
            fill=tuple(color),
            outline=tuple(color),
        )

    def convert_moge_to_delta_format(self, moge_points, mask, height, width):
        """Convert MoGe tracking format to DELTA format
        
        Args:
            moge_points (numpy.ndarray): Points array of shape [T, H, W, 3] with normalized coords
            mask (numpy.ndarray): Binary mask of shape [H, W]
            height (int): Frame height
            width (int): Frame width
            
        Returns:
            tuple: (delta_points, vis_mask)
                - delta_points (torch.Tensor): Points of shape [T, N, 3] with pixel coordinates
                - vis_mask (numpy.ndarray): Visibility mask of shape [T, N]
        """
        T, H, W, _ = moge_points.shape
        
        # Convert normalized coordinates to pixel coordinates
        pixel_points = moge_points.copy()
        pixel_points[:, :, :, 0] *= width   # x coordinates
        pixel_points[:, :, :, 1] *= height  # y coordinates
        # z coordinates remain the same
        
        # Flatten spatial dimensions and apply mask
        points_reshaped = pixel_points.reshape(T, H * W, 3)
        mask_flat = mask.flatten()
        
        # Only keep points that are in the mask
        valid_points = points_reshaped[:, mask_flat, :]
        
        # Create visibility mask (all points are visible by definition)
        N = valid_points.shape[1]
        vis_mask = np.ones((T, N), dtype=bool)
        
        # Convert to tensor
        delta_points = torch.from_numpy(valid_points).float()
        
        return delta_points, vis_mask

    ##============= DELTA =============##
    @torch.inference_mode()
    def predict_unidepth(self, video_torch, model):
        depth_pred = []
        chunks = torch.split(video_torch, 32, dim=0)
        for chunk in chunks:
            predictions = model.infer(chunk)
            depth_pred_ = predictions["depth"].squeeze(1).cpu().numpy()
            depth_pred.append(depth_pred_)
        depth_pred = np.concatenate(depth_pred, axis=0)

        return depth_pred

    @torch.inference_mode()
    def predict_depthcrafter(self, video, pipe):
        import cv2
        import torch.nn.functional as F
        def read_video(video, max_res):

            original_height, original_width = video.shape[2:4]

            height = round(original_height / 64) * 64
            width = round(original_width / 64) * 64

            # resize the video if the height or width is larger than max_res
            if max(height, width) > max_res:
                scale = max_res / max(original_height, original_width)
                height = round(original_height * scale / 64) * 64
                width = round(original_width * scale / 64) * 64
                print(f"Resized dimensions: {height}x{width}")

            # Ensure dimensions are positive
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")

            frames = []

            for frame in video:
                frame_np = frame.cpu().numpy() if hasattr(frame, 'cpu') else np.array(frame)
                # Convert from (C, H, W) to (H, W, C) for OpenCV
                frame_np = frame_np.transpose(1, 2, 0)
                frame_resized = cv2.resize(frame_np, (width, height))
                frames.append(frame_resized.astype("float32") / 255.0)

            frames = np.array(frames)
            return frames, original_height, original_width

        frames, ori_h, ori_w = read_video(video, max_res=1024)
        res = pipe(
            frames,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=1.2,
            num_inference_steps=25,
            window_size=110,
            overlap=25,
            track_time=False,
        ).frames[0]

        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())

        res = F.interpolate(torch.from_numpy(res[:, None]), (ori_h, ori_w), mode="nearest").squeeze(1).numpy()

        return res

    def generate_tracking_DELTA(self, video_tensor, density=70):
        """Generate tracking video
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            
        Returns:
            str: Path to tracking video
        """
        print("Loading tracking models...")
        # Load tracking model
        print("Initializing DenseTrack3D model")
        model = DenseTrack3D(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(384, 512),
            upsample_factor=4
        )
        
        print(f"Loading checkpoint from {os.path.join(project_root, 'checkpoints/densetrack3d.pth')}")
        with open(os.path.join(project_root, 'checkpoints/densetrack3d.pth'), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        
        predictor = DensePredictor3D(model=model).to(self.device)
        predictor = predictor.eval().cuda()
        
        # Initialize UniDepth
        os.sys.path.append(os.path.abspath(os.path.join(project_root, "submodules/DELTA/submodules/UniDepth")))
        from unidepth.models import UniDepthV2
        from unidepth.utils import colorize, image_grid
        
        unidepth_model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14")
        unidepth_model = unidepth_model.eval().to(self.device)

        print("Running Unidepth")
        # Convert from [B,C,T,H,W] to [T,C,H,W] for UniDepth
        video_for_unidepth = video_tensor.squeeze(0).permute(1, 0, 2, 3)  # [T,C,H,W]
        video_for_unidepth =video_for_unidepth * 255
        videodepth = self.predict_unidepth(video_for_unidepth, unidepth_model)

        use_depthcrafter = False
        if use_depthcrafter:
            os.sys.path.append(os.path.abspath(os.path.join(project_root, "submodules/DELTA/submodules/DepthCrafter")))
            from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
            from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
            from diffusers.training_utils import set_seed
            os.sys.path.append(os.path.abspath(os.path.join(project_root, "submodules/DELTA")))
            from densetrack3d.models.geometry_utils import least_square_align

            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                "tencent/DepthCrafter",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            # load weights of other components from the provided checkpoint
            depth_crafter_pipe = DepthCrafterPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            depth_crafter_pipe.to("cuda")
            # enable attention slicing and xformers memory efficient attention
            try:
                depth_crafter_pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(e)
                print("Xformers is not enabled")
            depth_crafter_pipe.enable_attention_slicing()

            print("Run DepthCrafter")
            videodisp = self.predict_depthcrafter(video_for_unidepth, depth_crafter_pipe)

            videodepth = videodisp
            # Save first frame of depth as image for visualization
            import cv2
            first_depth_frame = videodepth[0]  # Get first frame
            # Normalize depth values to 0-255 for visualization
            depth_normalized = ((first_depth_frame - first_depth_frame.min()) / (first_depth_frame.max() - first_depth_frame.min()) * 255).astype('uint8')
            cv2.imwrite(os.path.join(self.output_dir, "depthcrafter_first_frame.png"), depth_normalized)
    
        videodepth = torch.from_numpy(videodepth).unsqueeze(1).cuda()[None].float()
        
        try:
            print("Running DenseTrack3D")
            # Convert video_tensor from [B,C,T,H,W] to [B,T,C,H,W] for DELTA predictor and move to device
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                out_dict = predictor(
                    video_tensor,
                    videodepth,
                    grid_query_frame=0,
                )
            downsample = density
            trajs_uv = out_dict["trajs_uv"] # B T N 2
            trajs_vis = out_dict["vis"] # B T N 1
            dense_reso = out_dict["dense_reso"] # 2（h, w）
            trajs_depth = out_dict["trajs_depth"] # B T N 1

            sparse_trajs_uv = rearrange(trajs_uv, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
            sparse_trajs_uv = sparse_trajs_uv[:, :, :: downsample, :: downsample]
            sparse_trajs_uv = rearrange(sparse_trajs_uv, "b t h w c -> b t (h w) c")

            sparse_trajs_vis = rearrange(trajs_vis, "b t (h w) -> b t h w", h=dense_reso[0], w=dense_reso[1])
            sparse_trajs_vis = sparse_trajs_vis[:, :, :: downsample, :: downsample]
            sparse_trajs_vis = rearrange(sparse_trajs_vis, "b t h w -> b t (h w)")


            sparse_trajs_depth = rearrange(trajs_depth, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
            sparse_trajs_depth = sparse_trajs_depth[:, :, :: downsample, :: downsample]
            sparse_trajs_depth = rearrange(sparse_trajs_depth, "b t h w c -> b t (h w) c")

            # Extract dimensions
            B, T, N, _ = sparse_trajs_uv.shape
            
            # Create output tensor with depth
            pred_tracks_with_depth = torch.zeros((B, T, N, 3), device=sparse_trajs_uv.device)
            pred_tracks_with_depth[:, :, :, :2] = sparse_trajs_uv  # Copy x,y coordinates
            
            # Reshape depths back to [B, T, N] and assign to output tensor
            pred_tracks_with_depth[:, :, :, 2] = sparse_trajs_depth[:, :, :, 0]

            pred_tracks = pred_tracks_with_depth.squeeze(0)
            pred_visibility = sparse_trajs_vis.squeeze(0)


            return pred_tracks, pred_visibility
            
        finally:
            # Clean up GPU memory
            del model, predictor, unidepth_model
            torch.cuda.empty_cache()

    def fun_visualize_tracking_with_depth(self, pred_tracks_with_depth, pred_visibility, height, width, point_wise=4, mask_video=None, generate_type='full_edit'):
        """Visualize tracking results with depth information"""
        # Move tensors to CPU and convert to numpy
        if isinstance(pred_tracks_with_depth, torch.Tensor):
            points = pred_tracks_with_depth.detach().cpu().numpy()
        
        vis_mask = None
        if pred_visibility is not None and isinstance(pred_visibility, torch.Tensor):
            vis_mask = pred_visibility.detach().cpu().numpy()
            if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
                vis_mask = vis_mask.squeeze(-1)
        elif pred_visibility is not None:
            vis_mask = pred_visibility
        
        T, N, _ = points.shape
        H, W = height, width
        
        if vis_mask is None:
            vis_mask = np.ones((T, N), dtype=bool)
        
        colors = np.zeros((N, 3), dtype=np.uint8)
        
        first_frame_pts = points[0]
        
        u_min, u_max = 0, W
        u_normalized = np.clip((first_frame_pts[:, 0] - u_min) / (u_max - u_min), 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)
        
        v_min, v_max = 0, H
        v_normalized = np.clip((first_frame_pts[:, 1] - v_min) / (v_max - v_min), 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)
        
        z_values = first_frame_pts[:, 2]
        if np.all(z_values == 0):
            colors[:, 2] = np.random.randint(0, 256, N, dtype=np.uint8)
        else:
            inv_z = 1 / (z_values + 1e-10)
            p2 = np.percentile(inv_z, 2)
            p98 = np.percentile(inv_z, 98)
            normalized_z = np.clip((inv_z - p2) / (p98 - p2 + 1e-10), 0, 1)
            colors[:, 2] = (normalized_z * 255).astype(np.uint8)
        
        frames = []
        
        for i in tqdm(range(T), desc="rendering frames"):
            pts_i = points[i]
            visibility = vis_mask[i]
            
            pixels, depths = pts_i[visibility, :2], pts_i[visibility, 2]
            # Filter out invalid coordinates before casting
            valid_coords = np.isfinite(pixels).all(axis=1)
            pixels = pixels[valid_coords]
            depths = depths[valid_coords]
            pixels = pixels.astype(int)
            
            in_frame = self.valid_mask(pixels, W, H) 
            pixels = pixels[in_frame]
            depths = depths[in_frame]
            frame_rgb = colors[visibility][valid_coords][in_frame]
            
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")
            
            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            sorted_rgb = frame_rgb[sort_index]
            
            for j in range(sorted_pixels.shape[0]):
                coord = (sorted_pixels[j, 0], sorted_pixels[j, 1])
                if self._should_draw_point(coord, mask_video, i, generate_type, W, H):
                    self.draw_rectangle(img, coord=coord, side_length=point_wise, color=sorted_rgb[j])
            
            frames.append(np.array(img))
        
        return frames

    def apply_cosine_positional_encoding(self, pred_tracks_with_depth, height, width, L=4):
        """Apply cosine positional encoding to tracking points with normalization
        
        Args:
            pred_tracks_with_depth (torch.Tensor): Points array of shape [T, N, 3]
            video_shape (tuple): Video shape (B, C, T, H, W) for getting H, W
            L (int): Number of encoding levels
            
        Returns:
            list: List of L encoded tracking tensors
        """
        # Extract video dimensions
        H, W = height, width
        
        # Get first frame points for normalization reference
        first_frame_pts = pred_tracks_with_depth[0].detach().cpu().numpy()
        
        # Extract x, y, z coordinates
        x_coords = pred_tracks_with_depth[:, :, 0]  # [T, N]
        y_coords = pred_tracks_with_depth[:, :, 1]  # [T, N]
        z_coords = pred_tracks_with_depth[:, :, 2]  # [T, N]
        
        # Normalize x coordinates to [0, 1] based on video width
        u_min, u_max = 0, W
        x_normalized = torch.clamp((x_coords - u_min) / (u_max - u_min), 0, 1)
        
        # Normalize y coordinates to [0, 1] based on video height
        v_min, v_max = 0, H
        y_normalized = torch.clamp((y_coords - v_min) / (v_max - v_min), 0, 1)
        
        # Handle z coordinates - convert to inverse depth and normalize
        z_values = z_coords
        if torch.all(z_values == 0):
            # If all z values are 0, use random normalization
            z_normalized = torch.rand_like(z_values)
        else:
            # Convert to inverse depth
            inv_z = 1 / (z_values + 1e-10)
            
            # Use percentile-based normalization like in demo_3d_viz.py
            inv_z_np = inv_z.detach().cpu().numpy()
            p2 = np.percentile(inv_z_np, 2)
            p98 = np.percentile(inv_z_np, 98)
            
            # Convert back to tensor and normalize
            p2_tensor = torch.tensor(p2, device=inv_z.device, dtype=inv_z.dtype)
            p98_tensor = torch.tensor(p98, device=inv_z.device, dtype=inv_z.dtype)
            z_normalized = torch.clamp((inv_z - p2_tensor) / (p98_tensor - p2_tensor + 1e-10), 0, 1)
        
        # Create normalized tracking tensor
        normalized_tracks = torch.zeros_like(pred_tracks_with_depth)
        normalized_tracks[:, :, 0] = x_normalized
        normalized_tracks[:, :, 1] = y_normalized
        normalized_tracks[:, :, 2] = z_normalized
        
        encoded_tracks_list = []
        
        for i in range(L):
            # Calculate encoding factor: 2^i * pi
            encoding_factor = (2 ** i) * np.pi
            
            # Apply cosine encoding to normalized coordinates
            encoded_tracks = torch.cos(encoding_factor * normalized_tracks)
            
            encoded_tracks_list.append(encoded_tracks)
        
        return encoded_tracks_list


    def _save_video_frames(self, frames, output_path, fps=None):
        """Helper method to save video frames to file"""
        if fps is None:
            fps = self.fps
        try:
            uint8_frames = [frame.astype(np.uint8) for frame in frames]
            clip = ImageSequenceClip(uint8_frames, fps=fps)
            clip.write_videofile(output_path, codec="libx264", fps=fps, logger=None)
            print(f"Video saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Warning: Failed to save video to {output_path}: {e}")
            return None
    
    def _convert_frames_to_tensor(self, frames):
        """Convert video frames to tensor format [B, C, T, H, W]"""
        return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0
    
    def _prepare_vis_mask(self, vis_mask, points_shape):
        """Prepare visibility mask with proper shape and type"""
        if vis_mask is None:
            T, N, _ = points_shape
            return np.ones((T, N), dtype=bool)
        
        if isinstance(vis_mask, torch.Tensor):
            vis_mask = vis_mask.detach().cpu().numpy()
        
        if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
            vis_mask = vis_mask.squeeze(-1)
        
        return vis_mask
    
    def _generate_colors_from_points(self, first_frame_points, num_points):
        """Generate colors based on first frame point coordinates"""
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        
        # Normalize and map u coordinate to red channel
        u_normalized = np.clip((first_frame_points[:, 0] + 1) / 2, 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)
        
        # Normalize and map v coordinate to green channel
        v_normalized = np.clip((first_frame_points[:, 1] + 1) / 2, 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)
        
        # Normalize and map z coordinate to blue channel
        z_normalized = np.clip((first_frame_points[:, 2] + 1) / 2, 0, 1)
        colors[:, 2] = (z_normalized * 255).astype(np.uint8)
        
        return colors
    
    def _render_cosine_encoded_frame(self, points_t, vis_mask_t, colors, height, width, mask_video=None, frame_idx=0, generate_type='full_edit'):
        """Render a single frame with cosine encoded points"""
        visibility = vis_mask_t
        
        pixels = points_t[visibility, :2]
        depths = points_t[visibility, 2]
        # Filter out invalid coordinates before casting
        valid_coords = np.isfinite(pixels).all(axis=1)
        pixels = pixels[valid_coords].astype(int)
        depths = depths[valid_coords]
        
        # Filter points within frame bounds
        in_frame = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) & 
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        pixels = pixels[in_frame]
        depths = depths[in_frame]
        frame_colors = colors[visibility][valid_coords][in_frame]
        
        img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB")
        
        if len(pixels) > 0:
            # Sort by depth (far to near)
            sort_indices = depths.argsort()[::-1]
            sorted_pixels = pixels[sort_indices]
            sorted_colors = frame_colors[sort_indices]
            
            draw = ImageDraw.Draw(img)
            for pixel, color in zip(sorted_pixels, sorted_colors):
                coord = (int(pixel[0]), int(pixel[1]))
                if self._should_draw_point(coord, mask_video, frame_idx, generate_type, width, height):
                    left_up = (coord[0] - 2, coord[1] - 2)
                    right_down = (coord[0] + 2, coord[1] + 2)
                    draw.rectangle([left_up, right_down], fill=tuple(color), outline=tuple(color))
        
        return np.array(img)
    
    def _visualize_cosine_encoded_tracking(self, encoded_tracks_list, original_points, vis_mask, height, width, save_tracking, mask_video=None, generate_type='full_edit'):
        """Visualize cosine encoded tracking results"""
        cos_video_dict = {}
        
        for i, encoded_tracks in enumerate(encoded_tracks_list):
            print(f"Processing encoding level {i} (factor: 2^{i} * π = {(2**i) * np.pi:.2f})")
            
            encoded_points = encoded_tracks.detach().cpu().numpy()
            T, N, _ = encoded_points.shape
            
            # Generate colors based on first frame
            colors = self._generate_colors_from_points(encoded_points[0], N)
            
            # Generate frames
            frames = []
            for t in tqdm(range(T), desc=f"Rendering level {i}", leave=False):
                frame = self._render_cosine_encoded_frame(
                    original_points[t], vis_mask[t], colors, height, width,
                    mask_video=mask_video, frame_idx=t, generate_type=generate_type
                )
                frames.append(frame)
            
            # Convert to tensor
            cos_video = self._convert_frames_to_tensor(frames).unsqueeze(0)
            cos_video_dict[i] = cos_video
            
            # Save video if requested
            if save_tracking:
                output_path = os.path.join(self.output_dir, f"delta_cos_i_{i}.mp4")
                self._save_video_frames(frames, output_path)
        
        return cos_video_dict
    
    def _visualize_depth_tracking(self, points, vis_mask, height, width, point_wise, save_tracking, mask_video=None, generate_type='full_edit'):
        """Visualize depth-colored tracking results"""
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        T, N, _ = points.shape
        frames = []
        colormap = matplotlib.colormaps["Spectral"]
        
        for t in tqdm(range(T), desc="Rendering sparse depth frames"):
            img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB")
            
            uv_t = points[t, :, :2]
            depth_t = points[t, :, 2]
            vis_t = vis_mask[t].astype(bool)
            
            visible_uv = uv_t[vis_t]
            visible_depth = depth_t[vis_t]
            
            if len(visible_uv) == 0:
                frames.append(np.array(img))
                continue
            
            # Normalize depth using percentiles
            p2, p98 = np.percentile(visible_depth, [2, 98])
            if p98 > p2:
                depth_clipped = np.clip(visible_depth, p2, p98)
                depth_normalized = (depth_clipped - p2) / (p98 - p2)
            else:
                depth_normalized = np.zeros_like(visible_depth)
            
            # Convert to colors
            colors = (colormap(depth_normalized, bytes=False)[:, :3] * 255).astype(np.uint8)
            
            # Sort by depth (far to near)
            sort_indices = np.argsort(visible_depth)[::-1]
            sorted_uv = visible_uv[sort_indices]
            sorted_colors = colors[sort_indices]
            
            # Draw points
            for uv, color in zip(sorted_uv, sorted_colors):
                if np.isfinite(uv[0]) and np.isfinite(uv[1]):
                    coord = (int(uv[0]), int(uv[1]))
                    if 0 <= coord[0] < width and 0 <= coord[1] < height:
                        if self._should_draw_point(coord, mask_video, t, generate_type, width, height):
                            self.draw_rectangle(img, coord=coord, side_length=point_wise, color=tuple(color))
            
            frames.append(np.array(img))
        
        depth_video = self._convert_frames_to_tensor(frames).unsqueeze(0)
        
        # Save video if requested
        depth_path = None
        if save_tracking:
            output_path = os.path.join(self.output_dir, "depth_video_delta.mp4")
            depth_path = self._save_video_frames(frames, output_path)
        
        return depth_video

    def _load_mask_video(self, mask_path, generate_type, num_frames, height, width):
        """Load and preprocess mask video for fg/bg tracking"""
        if generate_type not in ['foreground_edit', 'background_edit'] or mask_path is None:
            return None
        
        try:
            from FlexAM.utils.utils import get_maskvideo_to_video_latent
            mask_video_input = get_maskvideo_to_video_latent(
                mask_path, video_length=num_frames, sample_size=[height, width]
            )
            if mask_video_input is not None:
                # Convert [F, C, H, W] to [F, H, W] binary mask
                mask_video = mask_video_input.mean(dim=1) > 0.5
                if generate_type == 'background_edit':
                    mask_video = ~mask_video  # Invert for background tracking
                return mask_video.float().cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not load mask video from {mask_path}: {e}")
        return None

    def _should_draw_point(self, coord, mask_video, frame_idx, generate_type, width, height):
        """Check if point should be drawn based on mask filtering"""
        if mask_video is None or generate_type not in ['foreground_edit', 'background_edit']:
            return True
        
        x, y = coord
        if 0 <= x < width and 0 <= y < height:
            return mask_video[frame_idx, int(y), int(x)] > 0.5
        return False

    def visualize_tracking_DELTA(self, points, vis_mask=None, save_tracking=True, point_wise=4, height=480, width=720, cos_level=4, generate_type='full_edit', mask_path=None):
        """Visualize tracking results from DELTA
        
        Args:
            points (torch.Tensor): Points array of shape [T, N, 3]
            vis_mask (torch.Tensor, optional): Visibility mask of shape [T, N, 1] or [T, N]
            save_tracking (bool): Whether to save tracking videos
            point_wise (int): Size of points in visualization
            height (int): Frame height
            width (int): Frame width
            cos_level (int): Number of cosine encoding levels
            generate_type (str): Generation type - 'full_edit', 'foreground_edit', 'background_edit'
            mask_path (str): Path to mask video for fg/bg tracking
            
        Returns:
            tuple: (tracking_video, cos_video_dict, depth_video)
        """
        # Prepare visibility mask
        vis_mask = self._prepare_vis_mask(vis_mask, points.shape)
        
        # Load mask video for fg/bg tracking if needed
        mask_video = self._load_mask_video(mask_path, generate_type, points.shape[0], height, width)
        
        # 1. Generate basic tracking video
        tracking_frames = self.fun_visualize_tracking_with_depth(
            points, vis_mask, height, width, point_wise=point_wise, 
            mask_video=mask_video, generate_type=generate_type
        )
        tracking_video = self._convert_frames_to_tensor(tracking_frames).unsqueeze(0)
        
        # Save basic tracking video
        if save_tracking:
            output_path = os.path.join(self.output_dir, "tracking_video_delta.mp4")
            self._save_video_frames(tracking_frames, output_path)
        
        # 2. Generate cosine encoded tracking videos
        print(f"Applying cosine positional encoding with {cos_level} levels")
        encoded_tracks_list = self.apply_cosine_positional_encoding(points, height, width, cos_level)
        original_points = points.detach().cpu().numpy()
        
        cos_video_dict = self._visualize_cosine_encoded_tracking(
            encoded_tracks_list, original_points, vis_mask, height, width, save_tracking,
            mask_video=mask_video, generate_type=generate_type
        )
        
        # 3. Generate depth visualization
        depth_video = self._visualize_depth_tracking(
            points, vis_mask, height, width, point_wise, save_tracking,
            mask_video=mask_video, generate_type=generate_type
        )
        
        return tracking_video, cos_video_dict, depth_video

    def apply_tracking(self,  fps=16, tracking_tensor=None, cos_video_dict = None,   depth_video = None, cos_level = 4,  full_ref = None, inpaint_video = None, inpaint_video_mask = None, prompt=None, checkpoint_path=None,num_inference_steps=40,  height = 480, width = 720,  video_length = 81, density=10, seed=42):
        """Generate final video with motion transfer
        
        Args:
            video_tensor (torch.Tensor): Input video tensor [T,C,H,W]
            fps (float): Input video FPS
            tracking_tensor (torch.Tensor): Tracking video tensor [T,C,H,W]
            image_tensor (torch.Tensor): First frame tensor [C,H,W] to use for generation
            prompt (str): Generation prompt
            checkpoint_path (str): Path to model checkpoint
        """
        self.fps = fps


        # Generate final video
        final_output = os.path.join(os.path.abspath(self.output_dir), "result.mp4")
        self._infer(
            prompt=prompt,
            model_path=checkpoint_path,
            tracking_tensor=tracking_tensor,
            cos_video_dict = cos_video_dict,
            depth_video = depth_video,
            cos_level = cos_level,
            full_ref = full_ref,
            inpaint_video      = inpaint_video,
            inpaint_video_mask   = inpaint_video_mask,
            output_path=final_output,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,
            dtype=torch.bfloat16,
            fps=self.fps,
            height = height, width = width,
            video_length = video_length,
            density = density,
            seed = seed
        )
        print(f"Final video generated successfully at: {final_output}")

    def _set_object_motion(self, motion_type):
        """Set object motion type
        
        Args:
            motion_type (str): Motion direction ('up', 'down', 'left', 'right')
        """
        self.object_motion = motion_type
