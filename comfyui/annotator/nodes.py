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


class VideoToTrackingPredict:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "density": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}), # 轨迹点密度
            }
        }

    RETURN_TYPES = ("TRACKING_DATA", "TRACKING_DATA")
    RETURN_NAMES = ("pred_tracks", "pred_visibility")
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

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

    def process(self, input_video, density):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 模型路径处理
        possible_folders = ["Fun_Models/Wan2.2-Fun-5B-FLEXAM","Fun_Models/Third_Party", "checkpoints", "DELTA"]
        densetrack_path = "densetrack3d.pth"

        found_path = None
        for folder in possible_folders:
            candidate = os.path.join(folder_paths.models_dir, folder, densetrack_path)
            if os.path.exists(candidate):
                found_path = candidate
                break

        if found_path is None:
            local_checkpoint = os.path.join(project_root, 'checkpoints', 'Wan2.2-Fun-5B-FLEXAM', 'densetrack3d.pth')
            if os.path.exists(local_checkpoint):
                found_path = local_checkpoint
            else:
                print(f"Warning: 'densetrack3d.pth' not found in models dir. Trying default relative path.")
                found_path = local_checkpoint

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
        if isinstance(input_video, torch.Tensor):
            video_for_unidepth = input_video.permute(0, 3, 1, 2).to(device)
            video_tensor_delta = video_for_unidepth.unsqueeze(0)
        else:
            video_frames = np.array(input_video * 255, np.uint8)
            video_for_unidepth = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() / 255.0
            video_for_unidepth = video_for_unidepth.to(device)
            video_tensor_delta = video_for_unidepth.unsqueeze(0)

        F_len, C, H, W = video_for_unidepth.shape

        # 5. 生成深度图 (UniDepth)
        print("Running UniDepth...")
        video_input_unidepth_scaled = video_for_unidepth * 255.0
        videodepth = self.predict_unidepth(video_input_unidepth_scaled, unidepth_model)

        videodepth_tensor = torch.from_numpy(videodepth).unsqueeze(1).to(device).float().unsqueeze(0)

        # 6. 生成轨迹 (DenseTrack3D Inference)
        print("Running DenseTrack3D...")
        print(video_tensor_delta.shape)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            out_dict = predictor(
                video_tensor_delta,
                videodepth_tensor,
                grid_query_frame=0,
            )

        # Parse output
        trajs_uv = out_dict["trajs_uv"]
        trajs_vis = out_dict["vis"]
        dense_reso = out_dict["dense_reso"]
        trajs_depth = out_dict["trajs_depth"]

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

        pred_tracks = pred_tracks.squeeze(0)  # [T, N, 3]
        pred_visibility = sparse_trajs_vis.squeeze(0)  # [T, N]

        # Cleanup memory
        del model, predictor, unidepth_model
        torch.cuda.empty_cache()

        return (pred_tracks, pred_visibility)


class VideoToTrackingVisualize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "pred_tracks": ("TRACKING_DATA",),
                "pred_visibility": ("TRACKING_DATA",),
                "point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
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
        combined = np.hstack((points, depths[:, None]))
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
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

    def process(self, input_video, pred_tracks, pred_visibility, point_size):
        # Get video dimensions
        if isinstance(input_video, torch.Tensor):
            F, H, W, C = input_video.shape
        else:
            video_frames = np.array(input_video)
            F, H, W, C = video_frames.shape

        # Convert tracking data to numpy
        points = pred_tracks.detach().cpu().numpy()
        vis_mask = pred_visibility.detach().cpu().numpy()
        if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
            vis_mask = vis_mask.squeeze(-1)

        T, N, _ = points.shape

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

        # Convert back to ComfyUI Tensor format: [F, H, W, C], float 0-1
        output_tensor = torch.from_numpy(np.array(output_frames)).float() / 255.0

        return (output_tensor,)


class VideoToCosVisualize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "pred_tracks": ("TRACKING_DATA",),
                "pred_visibility": ("TRACKING_DATA",),
                "point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "cos_level": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("cos_level_0", "cos_level_1", "cos_level_2", "cos_level_3")
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds"""
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values for Z-buffering"""
        combined = np.hstack((points, depths[:, None]))
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
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

    def apply_cosine_positional_encoding(self, points, height, width, cos_level=4):
        """Apply cosine positional encoding to tracking points

        Args:
            points: torch.Tensor [T, N, 3] - tracking points
            height, width: int - video dimensions
            cos_level: int - number of encoding levels

        Returns:
            list of encoded_tracks for each level
        """
        T, N, _ = points.shape

        # Extract x, y, z coordinates
        x_coords = points[:, :, 0]  # [T, N]
        y_coords = points[:, :, 1]  # [T, N]
        z_coords = points[:, :, 2]  # [T, N]

        # Normalize x coordinates to [0, 1] based on video width
        u_min, u_max = 0, width
        x_normalized = torch.clamp((x_coords - u_min) / (u_max - u_min), 0, 1)

        # Normalize y coordinates to [0, 1] based on video height
        v_min, v_max = 0, height
        y_normalized = torch.clamp((y_coords - v_min) / (v_max - v_min), 0, 1)

        # Handle z coordinates - convert to inverse depth and normalize
        z_values = z_coords
        if torch.all(z_values == 0):
            # If all z values are 0, use random normalization
            z_normalized = torch.rand_like(z_values)
        else:
            # Convert to inverse depth
            inv_z = 1 / (z_values + 1e-10)

            # Use percentile-based normalization
            inv_z_np = inv_z.detach().cpu().numpy()
            p2 = np.percentile(inv_z_np, 2)
            p98 = np.percentile(inv_z_np, 98)

            # Convert back to tensor and normalize
            p2_tensor = torch.tensor(p2, device=inv_z.device, dtype=inv_z.dtype)
            p98_tensor = torch.tensor(p98, device=inv_z.device, dtype=inv_z.dtype)
            z_normalized = torch.clamp((inv_z - p2_tensor) / (p98_tensor - p2_tensor + 1e-10), 0, 1)

        # Create normalized tracking tensor
        normalized_tracks = torch.zeros_like(points)
        normalized_tracks[:, :, 0] = x_normalized
        normalized_tracks[:, :, 1] = y_normalized
        normalized_tracks[:, :, 2] = z_normalized

        encoded_tracks_list = []

        for i in range(cos_level):
            # Calculate encoding factor: 2^i * pi
            encoding_factor = (2 ** i) * np.pi

            # Apply cosine encoding to normalized coordinates
            encoded_tracks = torch.cos(encoding_factor * normalized_tracks)

            encoded_tracks_list.append(encoded_tracks)

        return encoded_tracks_list

    def generate_colors_from_encoded_points(self, encoded_points, N):
        """Generate colors based on encoded cosine values"""
        colors = np.zeros((N, 3), dtype=np.uint8)

        # Map cosine values [-1, 1] to [0, 255]
        # Use np.clip to ensure values stay in valid range
        u_normalized = np.clip((encoded_points[:, 0] + 1) / 2, 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)

        v_normalized = np.clip((encoded_points[:, 1] + 1) / 2, 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)

        # Normalize and map z coordinate to blue channel
        z_normalized = np.clip((encoded_points[:, 2] + 1) / 2, 0, 1)
        colors[:, 2] = (z_normalized * 255).astype(np.uint8)

        return colors

    def process(self, input_video, pred_tracks, pred_visibility, point_size, cos_level):
        # Get video dimensions
        if isinstance(input_video, torch.Tensor):
            F, H, W, C = input_video.shape
        else:
            video_frames = np.array(input_video)
            F, H, W, C = video_frames.shape

        # Convert tracking data to numpy
        points = pred_tracks.detach().cpu().numpy()
        vis_mask = pred_visibility.detach().cpu().numpy()
        if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
            vis_mask = vis_mask.squeeze(-1)

        T, N, _ = points.shape

        # Apply cosine positional encoding
        encoded_tracks_list = self.apply_cosine_positional_encoding(
            pred_tracks, H, W, cos_level=min(cos_level, 4)
        )

        # Generate cos videos for each level
        cos_videos = []
        for i in range(4):  # Always return 4 outputs
            if i < len(encoded_tracks_list):
                encoded_points = encoded_tracks_list[i].detach().cpu().numpy()

                # Generate colors based on encoded values
                colors = self.generate_colors_from_encoded_points(encoded_points[0], N)

                output_frames = []
                for t in range(T):
                    pts_t = points[t]  # Use original points for position
                    visibility = vis_mask[t] > 0

                    # Filter valid points
                    pixels = pts_t[visibility, :2]
                    depths = pts_t[visibility, 2]

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

                # Convert to tensor
                output_tensor = torch.from_numpy(np.array(output_frames)).float() / 255.0
                cos_videos.append(output_tensor)
            else:
                # Return empty video for unused levels
                empty_video = torch.zeros((T, H, W, 3), dtype=torch.float32)
                cos_videos.append(empty_video)

        return tuple(cos_videos)


class VideoTodepthVisualize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "pred_tracks": ("TRACKING_DATA",),
                "pred_visibility": ("TRACKING_DATA",),
                "point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_video",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds"""
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

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

    def process(self, input_video, pred_tracks, pred_visibility, point_size):
        # Get video dimensions
        if isinstance(input_video, torch.Tensor):
            F, H, W, C = input_video.shape
        else:
            video_frames = np.array(input_video)
            F, H, W, C = video_frames.shape

        # Convert tracking data to numpy
        points = pred_tracks.detach().cpu().numpy()
        vis_mask = pred_visibility.detach().cpu().numpy()
        if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
            vis_mask = vis_mask.squeeze(-1)

        T, N, _ = points.shape

        # Use Spectral colormap for depth visualization
        colormap = matplotlib.colormaps["Spectral"]

        output_frames = []
        for t in range(T):
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")

            uv_t = points[t, :, :2]
            depth_t = points[t, :, 2]
            vis_t = vis_mask[t].astype(bool)

            visible_uv = uv_t[vis_t]
            visible_depth = depth_t[vis_t]

            if len(visible_uv) == 0:
                output_frames.append(np.array(img))
                continue

            # Normalize depth using percentiles
            p2, p98 = np.percentile(visible_depth, [2, 98])
            if p98 > p2:
                depth_clipped = np.clip(visible_depth, p2, p98)
                depth_normalized = (depth_clipped - p2) / (p98 - p2)
            else:
                depth_normalized = np.zeros_like(visible_depth)

            # Convert to colors using Spectral colormap
            colors = (colormap(depth_normalized, bytes=False)[:, :3] * 255).astype(np.uint8)

            # Sort by depth (far to near for proper occlusion)
            sort_indices = np.argsort(visible_depth)[::-1]
            sorted_uv = visible_uv[sort_indices]
            sorted_colors = colors[sort_indices]

            # Draw points
            for uv, color in zip(sorted_uv, sorted_colors):
                if np.isfinite(uv[0]) and np.isfinite(uv[1]):
                    coord = (int(uv[0]), int(uv[1]))
                    if 0 <= coord[0] < W and 0 <= coord[1] < H:
                        self.draw_rectangle(img, coord=coord, side_length=point_size, color=tuple(color))

            output_frames.append(np.array(img))

        # Convert to ComfyUI tensor format: [F, H, W, C], float 0-1
        output_tensor = torch.from_numpy(np.array(output_frames)).float() / 255.0

        return (output_tensor,)


class VideoToTrackingVisualizeAll:
    """Combined node that generates all tracking visualizations (tracking, cos, depth) with optional mask support"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "pred_tracks": ("TRACKING_DATA",),
                "pred_visibility": ("TRACKING_DATA",),
                "point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "cos_level": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "generate_type": (["motion_transfer", "fg_generation", "bg_generation"], {"default": "motion_transfer"}),
            },
            "optional": {
                "mask_video": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("tracking_video", "depth_video", "cos_level_0", "cos_level_1", "cos_level_2", "cos_level_3")
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds"""
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values for Z-buffering"""
        combined = np.hstack((points, depths[:, None]))
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
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

    def _load_mask_video(self, mask_video, generate_type, num_frames, height, width):
        """Load and preprocess mask video for fg/bg tracking"""
        if generate_type not in ['fg_generation', 'bg_generation'] or mask_video is None:
            return None

        try:
            # Convert mask_video to the format needed
            if isinstance(mask_video, torch.Tensor):
                mask_video_np = mask_video.cpu().numpy()  # [F, H, W, C]
            else:
                mask_video_np = np.array(mask_video)

            # Convert [F, H, W, C] to [F, H, W] binary mask
            if mask_video_np.ndim == 4:
                mask_video_np = mask_video_np.mean(axis=-1)  # Average across channels

            # Threshold to create binary mask
            mask_binary = mask_video_np > 0.5

            if generate_type == 'bg_generation':
                mask_binary = ~mask_binary  # Invert for background tracking

            return mask_binary.astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not load mask video: {e}")
            return None

    def _should_draw_point(self, coord, mask_video, frame_idx, generate_type, width, height):
        """Check if point should be drawn based on mask filtering"""
        if mask_video is None or generate_type not in ['fg_generation', 'bg_generation']:
            return True

        x, y = coord
        if 0 <= x < width and 0 <= y < height:
            return mask_video[frame_idx, int(y), int(x)] > 0.5
        return False

    def generate_colors_from_points(self, first_frame_points, num_points, height, width):
        """Generate colors based on first frame point coordinates"""
        colors = np.zeros((num_points, 3), dtype=np.uint8)

        # Normalize and map u coordinate to red channel
        u_min, u_max = 0, width
        u_normalized = np.clip((first_frame_points[:, 0] - u_min) / (u_max - u_min), 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)

        # Normalize and map v coordinate to green channel
        v_min, v_max = 0, height
        v_normalized = np.clip((first_frame_points[:, 1] - v_min) / (v_max - v_min), 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)

        # Handle z coordinate for blue channel
        z_values = first_frame_points[:, 2]
        if np.all(z_values == 0):
            colors[:, 2] = np.random.randint(0, 256, num_points, dtype=np.uint8)
        else:
            inv_z = 1 / (z_values + 1e-10)
            p2 = np.percentile(inv_z, 2)
            p98 = np.percentile(inv_z, 98)
            normalized_z = np.clip((inv_z - p2) / (p98 - p2 + 1e-10), 0, 1)
            colors[:, 2] = (normalized_z * 255).astype(np.uint8)

        return colors

    def apply_cosine_positional_encoding(self, points, height, width, cos_level=4):
        """Apply cosine positional encoding to tracking points - same as VideoToCosVisualize"""
        T, N, _ = points.shape

        # Extract x, y, z coordinates
        x_coords = points[:, :, 0]
        y_coords = points[:, :, 1]
        z_coords = points[:, :, 2]

        # Normalize coordinates
        x_normalized = torch.clamp((x_coords - 0) / width, 0, 1)
        y_normalized = torch.clamp((y_coords - 0) / height, 0, 1)

        # Handle z coordinates - convert to inverse depth and normalize
        z_values = z_coords
        if torch.all(z_values == 0):
            z_normalized = torch.rand_like(z_values)
        else:
            inv_z = 1 / (z_values + 1e-10)
            inv_z_np = inv_z.detach().cpu().numpy()
            p2 = np.percentile(inv_z_np, 2)
            p98 = np.percentile(inv_z_np, 98)
            p2_tensor = torch.tensor(p2, device=inv_z.device, dtype=inv_z.dtype)
            p98_tensor = torch.tensor(p98, device=inv_z.device, dtype=inv_z.dtype)
            z_normalized = torch.clamp((inv_z - p2_tensor) / (p98_tensor - p2_tensor + 1e-10), 0, 1)

        # Create normalized tracking tensor
        normalized_tracks = torch.zeros_like(points)
        normalized_tracks[:, :, 0] = x_normalized
        normalized_tracks[:, :, 1] = y_normalized
        normalized_tracks[:, :, 2] = z_normalized

        encoded_tracks_list = []
        for i in range(cos_level):
            encoding_factor = (2 ** i) * np.pi
            encoded_tracks = torch.cos(encoding_factor * normalized_tracks)
            encoded_tracks_list.append(encoded_tracks)

        return encoded_tracks_list

    def generate_colors_from_encoded_points(self, encoded_points, N):
        """Generate colors based on encoded cosine values"""
        colors = np.zeros((N, 3), dtype=np.uint8)
        u_normalized = np.clip((encoded_points[:, 0] + 1) / 2, 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)
        v_normalized = np.clip((encoded_points[:, 1] + 1) / 2, 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)
        z_normalized = np.clip((encoded_points[:, 2] + 1) / 2, 0, 1)
        colors[:, 2] = (z_normalized * 255).astype(np.uint8)
        return colors

    def process(self, input_video, pred_tracks, pred_visibility, point_size, cos_level, generate_type, mask_video=None):
        # Get video dimensions
        if isinstance(input_video, torch.Tensor):
            F, H, W, C = input_video.shape
        else:
            video_frames = np.array(input_video)
            F, H, W, C = video_frames.shape

        # Convert tracking data to numpy
        points = pred_tracks.detach().cpu().numpy()
        vis_mask = pred_visibility.detach().cpu().numpy()
        if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
            vis_mask = vis_mask.squeeze(-1)

        T, N, _ = points.shape

        # Load mask video if provided
        mask_video_processed = self._load_mask_video(mask_video, generate_type, T, H, W)

        # 1. Generate basic tracking video
        colors_tracking = self.generate_colors_from_points(points[0], N, H, W)
        tracking_frames = []

        for i in range(T):
            pts_i = points[i]
            visibility = vis_mask[i] > 0

            pixels = pts_i[visibility, :2]
            depths = pts_i[visibility, 2]

            valid_coords = np.isfinite(pixels).all(axis=1)
            pixels = pixels[valid_coords]
            depths = depths[valid_coords]
            pixels = pixels.astype(int)

            in_frame = self.valid_mask(pixels, W, H)
            pixels = pixels[in_frame]
            depths = depths[in_frame]

            frame_rgb = colors_tracking[visibility][valid_coords][in_frame]

            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")

            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            sorted_rgb = frame_rgb[sort_index]

            for j in range(sorted_pixels.shape[0]):
                coord = (sorted_pixels[j, 0], sorted_pixels[j, 1])
                if self._should_draw_point(coord, mask_video_processed, i, generate_type, W, H):
                    self.draw_rectangle(img, coord=coord, side_length=point_size, color=sorted_rgb[j])

            tracking_frames.append(np.array(img))

        tracking_tensor = torch.from_numpy(np.array(tracking_frames)).float() / 255.0 

        # 2. Generate depth video
        colormap = matplotlib.colormaps["Spectral"]
        depth_frames = []

        for t in range(T):
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")

            uv_t = points[t, :, :2]
            depth_t = points[t, :, 2]
            vis_t = vis_mask[t].astype(bool)

            visible_uv = uv_t[vis_t]
            visible_depth = depth_t[vis_t]

            if len(visible_uv) == 0:
                depth_frames.append(np.array(img))
                continue

            p2, p98 = np.percentile(visible_depth, [2, 98])
            if p98 > p2:
                depth_clipped = np.clip(visible_depth, p2, p98)
                depth_normalized = (depth_clipped - p2) / (p98 - p2)
            else:
                depth_normalized = np.zeros_like(visible_depth)

            colors = (colormap(depth_normalized, bytes=False)[:, :3] * 255).astype(np.uint8)

            sort_indices = np.argsort(visible_depth)[::-1]
            sorted_uv = visible_uv[sort_indices]
            sorted_colors = colors[sort_indices]

            for uv, color in zip(sorted_uv, sorted_colors):
                if np.isfinite(uv[0]) and np.isfinite(uv[1]):
                    coord = (int(uv[0]), int(uv[1]))
                    if 0 <= coord[0] < W and 0 <= coord[1] < H:
                        if self._should_draw_point(coord, mask_video_processed, t, generate_type, W, H):
                            self.draw_rectangle(img, coord=coord, side_length=point_size, color=tuple(color))

            depth_frames.append(np.array(img))

        depth_tensor = torch.from_numpy(np.array(depth_frames)).float() / 255.0

        # 3. Generate cosine encoded videos
        encoded_tracks_list = self.apply_cosine_positional_encoding(
            pred_tracks, H, W, cos_level=min(cos_level, 4)
        )

        cos_videos = []
        for i in range(4):
            if i < len(encoded_tracks_list):
                encoded_points = encoded_tracks_list[i].detach().cpu().numpy()
                colors_cos = self.generate_colors_from_encoded_points(encoded_points[0], N)

                cos_frames = []
                for t in range(T):
                    pts_t = points[t]
                    visibility = vis_mask[t] > 0

                    pixels = pts_t[visibility, :2]
                    depths = pts_t[visibility, 2]

                    valid_coords = np.isfinite(pixels).all(axis=1)
                    pixels = pixels[valid_coords]
                    depths = depths[valid_coords]
                    pixels = pixels.astype(int)

                    in_frame = self.valid_mask(pixels, W, H)
                    pixels = pixels[in_frame]
                    depths = depths[in_frame]

                    frame_rgb = colors_cos[visibility][valid_coords][in_frame]

                    img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")

                    sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
                    sorted_rgb = frame_rgb[sort_index]

                    for j in range(sorted_pixels.shape[0]):
                        coord = (sorted_pixels[j, 0], sorted_pixels[j, 1])
                        if self._should_draw_point(coord, mask_video_processed, t, generate_type, W, H):
                            self.draw_rectangle(img, coord=coord, side_length=point_size, color=sorted_rgb[j])

                    cos_frames.append(np.array(img))

                cos_tensor = torch.from_numpy(np.array(cos_frames)).float() / 255.0
                cos_videos.append(cos_tensor)
            else:
                empty_video = torch.zeros((T, H, W, 3), dtype=torch.float32)
                cos_videos.append(empty_video)

        return (tracking_tensor, depth_tensor, cos_videos[0], cos_videos[1], cos_videos[2], cos_videos[3])