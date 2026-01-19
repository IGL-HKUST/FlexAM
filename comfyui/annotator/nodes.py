# This folder is modified from the https://github.com/Mikubill/sd-webui-controlnet
import os

import cv2
import folder_paths
import numpy as np
import torch
from einops import rearrange

from .dwpose_utils import DWposeDetector
from .zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoe.zoedepth.utils.config import get_config

remote_onnx_det = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
remote_onnx_pose = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
remote_zoe= "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"



import matplotlib

from ...submodules.DELTA.densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from ...submodules.DELTA.densetrack3d.models.predictor.dense_predictor import DensePredictor3D
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
unidepth_root_path = os.path.join(current_dir, "UniDepth")
if unidepth_root_path not in sys.path:
    sys.path.insert(0, unidepth_root_path)
try:
    from unidepth.models import UniDepthV2
except ImportError as e:
    print(f"Failed to import UniDepth from {unidepth_root_path}: {e}")
from PIL import Image, ImageDraw


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
    cap.release()
    return frames

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def load_file_from_url(
    url: str,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    hash_prefix: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    from urllib.parse import urlparse
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress, hash_prefix=hash_prefix)
    return cached_file

class VideoToCanny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 81, "step": 4}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, input_video, low_threshold, high_threshold, video_length):
        def extract_canny_frames(frames):
            canny_frames = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, low_threshold, high_threshold)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                canny_frames.append(edges_colored)
            return canny_frames
        
        if type(input_video) is str:
            video_frames = read_video(input_video)
        else:
            video_frames = np.array(input_video * 255, np.uint8)[:video_length]
        output_video = extract_canny_frames(video_frames)
        output_video = torch.from_numpy(np.array(output_video)) / 255
        return (output_video,)

class VideoToDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 81, "step": 4}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"


    def process_frame(self, model, image, device, weight_dtype):
        with torch.no_grad():
            image, remove_pad = resize_image_with_pad(image, 512)
            image_depth = image
            with torch.no_grad():
                image_depth = torch.from_numpy(image_depth).to(device, weight_dtype)
                image_depth = image_depth / 255.0
                image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
                depth = model.infer(image_depth)

                depth = depth[0, 0].cpu().numpy()

                vmin = np.percentile(depth, 2)
                vmax = np.percentile(depth, 85)

                depth -= vmin
                depth /= vmax - vmin
                depth = 1.0 - depth
                depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
            image = remove_pad(depth_image)
            image = HWC3(image)
        return image

    def process(self, input_video, video_length):
        model = ZoeDepth.build_from_config(get_config("zoedepth", "infer"))

        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun/Third_Party", "Fun_Models/Third_Party", "VideoX_Fun/Third_Party"]  # Possible folder names to check

        # Check if the model exists in any of the possible folders within folder_paths.models_dir
        zoe_model_path = "ZoeD_M12_N.pt"
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, zoe_model_path)
            if os.path.exists(candidate_path):
                zoe_model_path = candidate_path
                break
        if not os.path.exists(zoe_model_path):
            load_file_from_url(remote_zoe, model_dir=os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party"))
            zoe_model_path = os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party", zoe_model_path)

        model.load_state_dict(
            torch.load(zoe_model_path, map_location="cpu")['model'], 
            strict=False
        )
        if torch.cuda.is_available():
            device = "cuda"
            weight_dtype = torch.float32
        else:
            device = "cpu"
            weight_dtype = torch.float32
        model = model.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)

        if isinstance(input_video, str):
            video_frames = read_video(input_video)
        else:
            video_frames = np.array(input_video * 255, np.uint8)[:video_length]

        output_video = [self.process_frame(model, frame, device, weight_dtype) for frame in video_frames]
        output_video = torch.from_numpy(np.array(output_video)) / 255

        return (output_video,)
    

class VideoToPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 81, "step": 4}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process_frame(self, model, image):
        with torch.no_grad():
            image, remove_pad = resize_image_with_pad(image, 512)
            pose_image = model(image)
            image = remove_pad(pose_image)
            image = HWC3(image)
        return image
    
    def process(self, input_video, video_length):
        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun/Third_Party", "Fun_Models/Third_Party", "VideoX_Fun/Third_Party"]  # Possible folder names to check

        # Check if the model exists in any of the possible folders within folder_paths.models_dir
        onnx_det = "yolox_l.onnx"
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, onnx_det)
            if os.path.exists(candidate_path):
                onnx_det = candidate_path
                break
        if not os.path.exists(onnx_det):
            load_file_from_url(remote_onnx_det, os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party"))
            onnx_det = os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party", onnx_det)
            
        onnx_pose = "dw-ll_ucoco_384.onnx"
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, onnx_pose)
            if os.path.exists(candidate_path):
                onnx_pose = candidate_path
                break
        if not os.path.exists(onnx_pose):
            load_file_from_url(remote_onnx_pose, os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party"))
            onnx_pose = os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party", onnx_pose)
        
        model = DWposeDetector(onnx_det, onnx_pose)

        if isinstance(input_video, str):
            video_frames = read_video(input_video)
        else:
            video_frames = np.array(input_video * 255, np.uint8)[:video_length]

        output_video = [self.process_frame(model, frame) for frame in video_frames]
        output_video = torch.from_numpy(np.array(output_video)) / 255
        return (output_video,)


class VideoToTracking:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "density": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}), # 轨迹点密度
                "point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}), # 可视化点的大小
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tracking_video",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds"""
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values for Z-buffering"""
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
        """Draw a rectangle on the PIL image"""
        draw = ImageDraw.Draw(rgb)
        left_up_point = (coord[0] - side_length//2, coord[1] - side_length//2)  
        right_down_point = (coord[0] + side_length//2, coord[1] + side_length//2)
        color = tuple(list(color))

        draw.rectangle(
            [left_up_point, right_down_point],
            fill=tuple(color),
            outline=tuple(color),
        )

    def predict_unidepth(self, video_torch, model):
        """Run UniDepth inference"""
        depth_pred = []
        chunks = torch.split(video_torch, 32, dim=0)
        for chunk in chunks:
            predictions = model.infer(chunk)
            depth_pred_ = predictions["depth"].squeeze(1).cpu().numpy()
            depth_pred.append(depth_pred_)
        depth_pred = np.concatenate(depth_pred, axis=0)
        return depth_pred

    def process(self, input_video, density, point_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 模型路径处理
        # 尝试在 ComfyUI 的 models 目录下寻找模型
        possible_folders = ["Fun_Models/Third_Party", "checkpoints", "DELTA"]
        densetrack_path = "densetrack3d.pth"
        
        found_path = None
        for folder in possible_folders:
            candidate = os.path.join(folder_paths.models_dir, folder, densetrack_path)
            if os.path.exists(candidate):
                found_path = candidate
                break
        
        # 如果找不到，回退到当前插件目录下的 checkpoints (参考 pipelines.py 的逻辑)
        if found_path is None:
            local_checkpoint = os.path.join(project_root, 'checkpoints', 'densetrack3d.pth')
            if os.path.exists(local_checkpoint):
                found_path = local_checkpoint
            else:
                # 这里可以添加 download_url_to_file 逻辑，如果有 URL 的话
                print(f"Warning: 'densetrack3d.pth' not found in models dir. Trying default relative path.")
                found_path = local_checkpoint # 尝试硬闯，或者报错

        print(f"Loading DenseTrack3D from: {found_path}")

        # 2. 加载 DenseTrack3D 模型
        model = DenseTrack3D(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(384, 512),
            upsample_factor=4
        )
        
        if os.path.exists(found_path):
            with open(found_path, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
        else:
             raise FileNotFoundError(f"Could not find densetrack3d.pth at {found_path}")

        predictor = DensePredictor3D(model=model).to(device).eval()


        unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        unidepth_model = unidepth_model.eval().to(device)

        # 4. 数据预处理
        # ComfyUI input: [Frames, Height, Width, Channel] (F, H, W, 3) 范围 0-1
        # UniDepth need: [Frames, Channel, Height, Width] (F, 3, H, W)
        # DELTA need:    [Batch, Channel, Frames, Height, Width] (B, 3, F, H, W)
        
        if isinstance(input_video, torch.Tensor):
            # input_video is [F, H, W, C]
            # Convert to [F, C, H, W] for UniDepth
            video_for_unidepth = input_video.permute(0, 3, 1, 2).to(device)
            # DELTA expects [B, C, F, H, W] -> We add Batch dim
            video_tensor_delta = video_for_unidepth.unsqueeze(0) 
        else:
            # Fallback for list of frames if necessary, but Comfy usually gives Tensor
            video_frames = np.array(input_video * 255, np.uint8)
            video_for_unidepth = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() / 255.0
            video_for_unidepth = video_for_unidepth.to(device)
            video_tensor_delta = video_for_unidepth.unsqueeze(0)

        F_len, C, H, W = video_for_unidepth.shape

        # 5. 生成深度图 (UniDepth)
        print("Running UniDepth...")
        # UniDepth expects input in range [0, 1] (or check model spec, usually generic float)
        # Pipelines.py multiplied by 255 before unidepth? 
        # Code check: `video_for_unidepth = video_for_unidepth * 255` in pipelines.py.
        # Let's verify input range. ComfyUI gives 0-1. 
        video_input_unidepth_scaled = video_for_unidepth * 255.0
        videodepth = self.predict_unidepth(video_input_unidepth_scaled, unidepth_model)
        
        # Format depth for DELTA: [B, 1, F, H, W] ?
        # Pipelines code: `videodepth = torch.from_numpy(videodepth).unsqueeze(1).cuda()[None].float()`
        # videodepth from predict is [F, H, W]. 
        # unsqueeze(1) -> [F, 1, H, W]. 
        # [None] -> [1, F, 1, H, W].
        # DELTA predictor expects `videodepth` as [B, F, 1, H, W] ?? 
        # Let's check pipelines.py logic:
        # video_tensor passed to predictor is [B, F, C, H, W] (after permute).
        # videodepth passed is [1, F, 1, H, W].
        
        videodepth_tensor = torch.from_numpy(videodepth).unsqueeze(1).to(device).float().unsqueeze(0) # [1, F, 1, H, W]
        
        # 6. 生成轨迹 (DenseTrack3D Inference)
        print("Running DenseTrack3D...")
        # Prepare video tensor for DELTA: [B, F, C, H, W]
        # video_tensor_delta = video_tensor_delta.permute(0, 2, 1, 3, 4) # [B, F, C, H, W] 
        print(video_tensor_delta.shape)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            out_dict = predictor(
                video_tensor_delta,
                videodepth_tensor,
                grid_query_frame=0,
            )
        
        # Parse output
        trajs_uv = out_dict["trajs_uv"] # B T N 2
        trajs_vis = out_dict["vis"] # B T N 1
        dense_reso = out_dict["dense_reso"] # (h, w)
        trajs_depth = out_dict["trajs_depth"] # B T N 1

        # Downsample / Density control
        downsample = density
        
        sparse_trajs_uv = rearrange(trajs_uv, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_uv = sparse_trajs_uv[:, :, :: downsample, :: downsample]
        sparse_trajs_uv = rearrange(sparse_trajs_uv, "b t h w c -> b t (h w) c")

        sparse_trajs_vis = rearrange(trajs_vis, "b t (h w) -> b t h w", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_vis = sparse_trajs_vis[:, :, :: downsample, :: downsample]
        sparse_trajs_vis = rearrange(sparse_trajs_vis, "b t h w -> b t (h w)")

        sparse_trajs_depth = rearrange(trajs_depth, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_depth = sparse_trajs_depth[:, :, :: downsample, :: downsample]
        sparse_trajs_depth = rearrange(sparse_trajs_depth, "b t h w c -> b t (h w) c")

        # Prepare for visualization
        B, T, N, _ = sparse_trajs_uv.shape
        pred_tracks = torch.zeros((B, T, N, 3), device=device)
        pred_tracks[:, :, :, :2] = sparse_trajs_uv
        pred_tracks[:, :, :, 2] = sparse_trajs_depth[:, :, :, 0]
        
        pred_tracks = pred_tracks.squeeze(0) # [T, N, 3]
        pred_visibility = sparse_trajs_vis.squeeze(0) # [T, N]

        # 7. 可视化 (Visualization)
        print("Visualizing Tracking Video...")
        
        points = pred_tracks.detach().cpu().numpy()
        vis_mask = pred_visibility.detach().cpu().numpy()
        if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
            vis_mask = vis_mask.squeeze(-1)

        # Generate colors based on first frame position
        colors = np.zeros((N, 3), dtype=np.uint8)
        first_frame_pts = points[0]
        
        u_min, u_max = 0, W
        u_normalized = np.clip((first_frame_pts[:, 0] - u_min) / (u_max - u_min), 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)
        
        v_min, v_max = 0, H
        v_normalized = np.clip((first_frame_pts[:, 1] - v_min) / (v_max - v_min), 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)
        
        # Z-Coloring
        z_values = first_frame_pts[:, 2]
        if np.all(z_values == 0):
            colors[:, 2] = np.random.randint(0, 256, N, dtype=np.uint8)
        else:
            inv_z = 1 / (z_values + 1e-10)
            p2 = np.percentile(inv_z, 2)
            p98 = np.percentile(inv_z, 98)
            normalized_z = np.clip((inv_z - p2) / (p98 - p2 + 1e-10), 0, 1)
            colors[:, 2] = (normalized_z * 255).astype(np.uint8)

        output_frames = []
        for i in range(T):
            pts_i = points[i]
            visibility = vis_mask[i] > 0
            
            # Filter valid points
            pixels = pts_i[visibility, :2]
            depths = pts_i[visibility, 2]
            
            valid_coords = np.isfinite(pixels).all(axis=1)
            pixels = pixels[valid_coords]
            depths = depths[valid_coords]
            pixels = pixels.astype(int)
            
            in_frame = self.valid_mask(pixels, W, H)
            pixels = pixels[in_frame]
            depths = depths[in_frame]
            
            # Filter colors
            frame_rgb = colors[visibility][valid_coords][in_frame]
            
            # Create blank image
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")
            
            # Sort by depth for correct occlusion
            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            sorted_rgb = frame_rgb[sort_index]
            
            # Draw
            for j in range(sorted_pixels.shape[0]):
                coord = (sorted_pixels[j, 0], sorted_pixels[j, 1])
                self.draw_rectangle(img, coord=coord, side_length=point_size, color=sorted_rgb[j])
            
            output_frames.append(np.array(img))

        # Cleanup memory
        del model, predictor, unidepth_model
        torch.cuda.empty_cache()

        # Convert back to ComfyUI Tensor format: [F, H, W, C], float 0-1
        output_tensor = torch.from_numpy(np.array(output_frames)).float() / 255.0
        
        return (output_tensor,)