import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from random import shuffle
from threading import Thread

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from scipy.ndimage import gaussian_filter

VIDEO_READER_TIMEOUT = 20

#### generate full tracking mask
def get_random_mask(shape, image_start_only=True):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

#### generate fg tracking mask
def generate_mask_fg_tracking_diable_bucket(mask_video_input):
    """
    Generate mask from mask video input.
    
    Args:
        mask_video_input: Video numpy array of shape [F, H, W, C] with pixel values
    
    Returns:
        mask: Binary mask tensor of shape [F, 1, H, W] where:
            - First frame is always 0
            - Black pixels (low values) -> 0 (not masked)
            - White pixels (high values) -> 1 (masked)
    """
    
    # Convert numpy array [F, H, W, C] to tensor [F, C, H, W]
    mask_video_input = torch.from_numpy(mask_video_input).permute(0, 3, 1, 2).contiguous().float()
    
    f, c, h, w = mask_video_input.shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)
    
    # First frame is always 0 (no mask)
    # Process subsequent frames
    for frame_idx in range(1, f):
        frame = mask_video_input[frame_idx]  # [C, H, W]
        
        # Convert to grayscale if needed (take mean across channels)
        if c > 1:
            gray_frame = frame.mean(dim=0, keepdim=True)  # [1, H, W]
        else:
            gray_frame = frame  # [1, H, W]
        
        # Normalize to 0-1 range and threshold
        # Assuming input is in range [0, 255] or similar
        normalized = gray_frame / 255.0 if gray_frame.max() > 1.0 else gray_frame
        
        # White pixels (> 0.5) become 1 (masked), black pixels (< 0.5) become 0 (not masked)
        binary_mask = (normalized > 0.5).float()
        
        mask[frame_idx] = binary_mask
    
    return mask

def generate_mask_fg_tracking_enable_bucket(mask_video_input, blur_radius: int = 15):
    """
    从 mask_video_input 生成 refined 二值 mask，处理流程：
    原始mask高斯模糊 → 凸包形状 → 像素扩展 → 二值化
    
    Args:
        mask_video_input: torch.Tensor [F, C, H, W]，数值范围 [0, 1] 或 [0,255]
        blur_radius: 高斯模糊半径
    
    Returns:
        mask: torch.Tensor [F, 1, H, W], dtype=torch.uint8, 值为 {0,1}
              - 第 0 帧始终全 0
    """
    f, c, h, w = mask_video_input.shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)  # 初始化全 0

    # 转灰度
    if c > 1:
        gray_frames = mask_video_input.mean(dim=1, keepdim=True)  # [F,1,H,W]
    else:
        gray_frames = mask_video_input
    # 随机从1-6取dilation_pixels值
    dilation_pixels = random.randint(1, 6)  # 生成1到6之间的随机整数
    for i in range(1, f):  # 跳过第 0 帧
        frame = gray_frames[i,0].cpu().numpy()  # [H,W]

        # 原始二值化
        mask_uint8 = (frame > 0.5).astype(np.uint8) * 255

        # Step 1: 高斯模糊 + 再二值化
        if blur_radius > 0:
            mask_float = mask_uint8.astype(np.float32) / 255.0
            blurred = gaussian_filter(mask_float, sigma=blur_radius/6.0)
            mask_uint8 = (blurred > 0.5).astype(np.uint8) * 255

        # Step 2: 凸包
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convex_mask = np.zeros_like(mask_uint8)
        for contour in contours:
            if len(contour) >= 3:
                hull = cv2.convexHull(contour)
                cv2.fillPoly(convex_mask, [hull], 255)

        # Step 3: 膨胀
        if dilation_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (dilation_pixels*2+1, dilation_pixels*2+1))
            convex_mask = cv2.dilate(convex_mask, kernel, iterations=1)

        # 转为 {0,1} uint8 并存入结果
        final_mask = (convex_mask > 127).astype(np.uint8)
        mask[i,0] = torch.from_numpy(final_mask)

    # 确保第 0 帧全 0
    mask[0,0] = 0

    return mask

def generate_mask_fg_tracking_for_validation(mask_video_input, blur_radius: int = 15):
    """
    从 mask_video_input 生成 refined 二值 mask，处理流程：
    原始mask高斯模糊 → 凸包形状 → 像素扩展 → 二值化
    
    Args:
        mask_video_input: torch.Tensor [F, C, H, W]，数值范围 [0, 1] 或 [0,255]
        blur_radius: 高斯模糊半径
    
    Returns:
        mask: torch.Tensor [F, 1, H, W], dtype=torch.uint8, 值为 {0,1}
              - 第 0 帧始终全 0
    """
    f, c, h, w = mask_video_input.shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)  # 初始化全 0

    # 转灰度
    if c > 1:
        gray_frames = mask_video_input.mean(dim=1, keepdim=True)  # [F,1,H,W]
    else:
        gray_frames = mask_video_input
    # 随机从1-6取dilation_pixels值
    dilation_pixels = random.randint(1, 6)  # 生成1到6之间的随机整数
    for i in range(1, f):  # 跳过第 0 帧
        frame = gray_frames[i,0].cpu().numpy()  # [H,W]

        # 原始二值化
        mask_uint8 = (frame > 0.5).astype(np.uint8) * 255

        # Step 1: 高斯模糊 + 再二值化
        if blur_radius > 0:
            mask_float = mask_uint8.astype(np.float32) / 255.0
            blurred = gaussian_filter(mask_float, sigma=blur_radius/6.0)
            mask_uint8 = (blurred > 0.5).astype(np.uint8) * 255

        # Step 2: 凸包
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convex_mask = np.zeros_like(mask_uint8)
        for contour in contours:
            if len(contour) >= 3:
                hull = cv2.convexHull(contour)
                cv2.fillPoly(convex_mask, [hull], 255)

        # Step 3: 膨胀
        if dilation_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (dilation_pixels*2+1, dilation_pixels*2+1))
            convex_mask = cv2.dilate(convex_mask, kernel, iterations=1)

        # 转为 {0,1} uint8 并存入结果
        final_mask = (convex_mask > 127).astype(np.uint8)
        mask[i,0] = torch.from_numpy(final_mask)

    # 确保第 0 帧全 0
    mask[0,0] = 0

    return mask

#### generate bg tracking mask
def generate_mask_bg_tracking_diable_bucket(mask_video_input):
    """
    Generate mask from mask video input.
    
    Args:
        mask_video_input: Video numpy array of shape [F, H, W, C] with pixel values
    
    Returns:
        mask: Binary mask tensor of shape [F, 1, H, W] where:
            - First frame is always 0
            - Black pixels (low values) -> 0 (not masked)
            - White pixels (high values) -> 1 (masked)
    """
    
    # Convert numpy array [F, H, W, C] to tensor [F, C, H, W]
    mask_video_input = torch.from_numpy(mask_video_input).permute(0, 3, 1, 2).contiguous().float()
    
    f, c, h, w = mask_video_input.shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)
    
    # First frame is always 0 (no mask)
    # Process subsequent frames
    for frame_idx in range(1, f):
        frame = mask_video_input[frame_idx]  # [C, H, W]
        
        # Convert to grayscale if needed (take mean across channels)
        if c > 1:
            gray_frame = frame.mean(dim=0, keepdim=True)  # [1, H, W]
        else:
            gray_frame = frame  # [1, H, W]
        
        # Normalize to 0-1 range and threshold
        # Assuming input is in range [0, 255] or similar
        normalized = gray_frame / 255.0 if gray_frame.max() > 1.0 else gray_frame
        
        # White pixels (> 0.5) become 1 (masked), black pixels (< 0.5) become 0 (not masked)
        binary_mask = (normalized < 0.5).float()
        
        mask[frame_idx] = binary_mask
    
    return mask

def generate_mask_bg_tracking_enable_bucket(mask_video_input):
    """
    Generate mask from mask video input.
    
    Args:
        mask_video_input: Video tensor of shape [F, C, H, W] with pixel values, normalize to [0, 1]
    
    Returns:
        mask: Binary mask tensor of shape [F, 1, H, W] where:
            - First frame is always 0
            - Black pixels (low values) -> 0 (not masked)
            - White pixels (high values) -> 1 (masked)
    """
    
    f, c, h, w = mask_video_input.shape
    
    # Convert to grayscale if needed (vectorized for all frames)
    if c > 1:
        gray_frames = mask_video_input.mean(dim=1, keepdim=True)  # [F, 1, H, W]
    else:
        gray_frames = mask_video_input  # [F, 1, H, W]
    
    # Create binary mask for all frames at once
    mask = (gray_frames < 0.5).to(torch.uint8)
    
    # First frame is always 0 (no mask)
    mask[0] = 0
    
    return mask

def generate_mask_bg_tracking_for_validation(mask_video_input):
    """
    Generate mask from mask video input.
    
    Args:
        mask_video_input: Video tensor of shape [F, C, H, W] with pixel values
    
    Returns:
        mask: Binary mask tensor of shape [F, 1, H, W] where:
            - First frame is always 0
            - Black pixels (low values) -> 0 (not masked)
            - White pixels (high values) -> 1 (masked)
    """
    
    f, c, h, w = mask_video_input.shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.float32)
    
    # First frame is always 0 (no mask)
    # Process subsequent frames
    for frame_idx in range(1, f):
        frame = mask_video_input[frame_idx]  # [C, H, W]
        
        # Convert to grayscale if needed (take mean across channels)
        if c > 1:
            gray_frame = frame.mean(dim=0, keepdim=True)  # [1, H, W]
        else:
            gray_frame = frame  # [1, H, W]
        
        # Normalize to 0-1 range and threshold
        # Assuming input is in range [0, 255] or similar
        normalized = gray_frame / 255.0 if gray_frame.max() > 1.0 else gray_frame
        
        # White pixels (> 0.5) become 1 (masked), black pixels (< 0.5) become 0 (not masked)
        binary_mask = (normalized < 0.5).float()
        
        mask[frame_idx] = binary_mask
    
    return mask

#### video color jitter
def adjust_brightness(img, factor):
    """调整亮度，输入输出均为[0,255]范围"""
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor):
    """调整对比度，输入输出均为[0,255]范围"""
    mean = img.mean(axis=(0,1), keepdims=True).astype(np.float32)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

def adjust_saturation(img, factor):
    """调整饱和度，输入输出均为[0,255]范围"""
    # 转换为灰度图
    gray = np.mean(img, axis=2, keepdims=True).astype(np.float32)
    # 调整饱和度
    return np.clip((img.astype(np.float32) - gray) * factor + gray, 0, 255).astype(np.uint8)

def adjust_hue(img, factor):
    """调整色调，输入输出均为[0,255]范围的RGB图像"""
    # 转换为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 调整色调 (H通道范围是0-179)
    hsv[..., 0] = (hsv[..., 0] + factor * 180) % 180
    # 转换回RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def video_color_jitter(video_array, 
                            brightness=0.2, 
                            contrast=0.2, 
                            saturation=0.2, 
                            hue=0.1):
    """
    对numpy视频数组应用颜色抖动，整个视频使用相同的参数
    
    Args:
        video_array: 视频数组，形状为 [F, H, W, C]，值范围 [0, 255]，RGB格式
        brightness: 亮度调整因子范围，实际因子为 [1-brightness, 1+brightness]
        contrast: 对比度调整因子范围，实际因子为 [1-contrast, 1+contrast]
        saturation: 饱和度调整因子范围，实际因子为 [1-saturation, 1+saturation]
        hue: 色调调整范围，实际调整为 [-hue, hue]
        
    Returns:
        经过颜色抖动的视频数组，形状与输入相同，值范围 [0, 255]
    """
    # 确保输入格式正确
    assert len(video_array.shape) == 4, "输入必须是[F, H, W, C]格式的数组"
    assert video_array.dtype == np.uint8, "输入数组必须是uint8类型"
    
    F, H, W, C = video_array.shape
    jittered_video = np.zeros_like(video_array)
    
    # 随机生成整个视频共用的抖动参数
    brightness_factor = random.uniform(1 - brightness, 1 + brightness)
    contrast_factor = random.uniform(1 - contrast, 1 + contrast)
    saturation_factor = random.uniform(1 - saturation, 1 + saturation)
    hue_factor = random.uniform(-hue, hue)
    
    # 对每一帧应用相同的抖动参数
    for f in range(F):
        frame = video_array[f]
        
        # 应用颜色抖动
        frame = adjust_brightness(frame, brightness_factor)
        frame = adjust_contrast(frame, contrast_factor)
        frame = adjust_saturation(frame, saturation_factor)
        frame = adjust_hue(frame, hue_factor)
        
        jittered_video[f] = frame
    
    return jittered_video

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

def process_pose_file(pose_file_path, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    with open(pose_file_path, 'r') as f:
        poses = f.readlines()

    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

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

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding

def process_pose_params(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu'):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

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

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding

class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

class ImageVideoControlDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=0.9,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        cos_level=4,
        apply_color_jitter=True
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
        self.cos_level = cos_level
        self.apply_color_jitter = apply_color_jitter

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']
        # generate_type
        generate_type = data_info.get('generate_type', 'full_tracking')
        if data_info.get('type', 'image')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")
                
                # if self.apply_color_jitter and generate_type == 'bg_tracking':
                #     pixel_values = video_color_jitter(pixel_values)

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''

            control_video_id = data_info['control_file_path']
            
            if control_video_id is not None:
                if self.data_root is None:
                    control_video_id = control_video_id
                else:
                    control_video_id = os.path.join(self.data_root, control_video_id)
                
            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                if control_video_id is not None:
                    with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            resized_frames = []
                            for i in range(len(control_pixel_values)):
                                frame = control_pixel_values[i]
                                resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                resized_frames.append(resized_frame)
                            control_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                            control_pixel_values = control_pixel_values / 255.
                            del control_video_reader
                        else:
                            control_pixel_values = control_pixel_values

                        if not self.enable_bucket:
                            control_pixel_values = self.video_transforms(control_pixel_values)
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                control_camera_values = None
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = pixel_values.shape[-2:]
                else:
                    visual_height, visual_width = pixel_values.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i])
                    width, height = subject_image.size
                    total_pixels = width * height

                    img = padding_image(subject_image, visual_width, visual_height)
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(img)
                subject_image = np.array(subject_images)
            else:
                subject_image = None


            ########################### Additional control signals ###########################
            
            # ================== Depth Video Processing ==================
            depth_video_id = data_info.get('depth_file_path')
            if isinstance(depth_video_id, str):
                depth_video_id = depth_video_id.strip()
                if depth_video_id == '':
                    depth_video_id = None

            if depth_video_id is not None:
                if self.data_root is not None and not os.path.isabs(depth_video_id):
                    depth_video_id = os.path.join(self.data_root, depth_video_id)
                with VideoReader_contextmanager(depth_video_id, num_threads=2) as depth_video_reader:
                    try:
                        sample_args = (depth_video_reader, batch_index)
                        depth_pixel_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for frame in depth_pixel_values:
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        depth_pixel_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from depth video. Error is {e}.")

                    if not self.enable_bucket:
                        depth_pixel_values = torch.from_numpy(depth_pixel_values).permute(0, 3, 1, 2).contiguous()
                        depth_pixel_values = depth_pixel_values / 255.
                        del depth_video_reader
                    else:
                        depth_pixel_values = depth_pixel_values

                    if not self.enable_bucket:
                        depth_pixel_values = self.video_transforms(depth_pixel_values)
            else:
                if not self.enable_bucket:
                    depth_pixel_values = torch.zeros_like(pixel_values)
                else:
                    depth_pixel_values = np.zeros_like(pixel_values)

            # ================== Mask Video Processing ==================
            mask_video_input = None
            
            if generate_type == 'full_tracking':
                mask_video_id = None
            else:
                mask_video_id = data_info.get('mask_file_path', '').strip()
                if not mask_video_id:
                    raise ValueError(f"mask_file_path is required for generate_type '{generate_type}'")

            if mask_video_id is not None:
                if self.data_root is not None and not os.path.isabs(mask_video_id):
                    mask_video_id = os.path.join(self.data_root, mask_video_id)
                with VideoReader_contextmanager(mask_video_id, num_threads=2) as mask_video_reader:
                    try:
                        mask_batch_index = batch_index.copy()
                        mask_video_length = len(mask_video_reader)
                        if mask_video_length == 0:
                            raise ValueError("Mask video contains no frames.")

                        if mask_batch_index.max() >= mask_video_length:
                            mask_batch_index = np.clip(mask_batch_index, 0, mask_video_length - 1)

                        sample_args = (mask_video_reader, mask_batch_index)
                        mask_video_input = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for frame in mask_video_input:
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        mask_video_input = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from mask video. Error is {e}.")


            # ================== CoS Video Processing ==================
            cos_pixel_values_list = [None] * self.cos_level
            
            # Collect cos paths from data_info
            cos_paths = [None] * self.cos_level
            for i in range(self.cos_level):
                cos_path = data_info.get(f'cos_{i}_file_path')
                if isinstance(cos_path, str) and cos_path.strip():
                    cos_paths[i] = cos_path.strip()

            # Auto-infer missing cos paths based on the first one (xxx_cos_i_{i}.mp4 pattern)
            if cos_paths[0]:
                for i in range(1, self.cos_level):
                    if cos_paths[i] is None:
                        cos_paths[i] = cos_paths[0].replace('_cos_i_0', f'_cos_i_{i}')

            # Process each cos video
            for i, cos_path in enumerate(cos_paths):
                if cos_path is None:
                    continue
                if self.data_root is not None and not os.path.isabs(cos_path):
                    cos_path = os.path.join(self.data_root, cos_path)
                with VideoReader_contextmanager(cos_path, num_threads=2) as cos_video_reader:
                    try:
                        sample_args = (cos_video_reader, batch_index)
                        cos_pixel_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for frame in cos_pixel_values:
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        cos_pixel_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from cos_{i} video. Error is {e}.")

                    if not self.enable_bucket:
                        cos_pixel_values = torch.from_numpy(cos_pixel_values).permute(0, 3, 1, 2).contiguous()
                        cos_pixel_values = cos_pixel_values / 255.
                        del cos_video_reader
                    else:
                        cos_pixel_values = cos_pixel_values

                    if not self.enable_bucket:
                        cos_pixel_values = self.video_transforms(cos_pixel_values)

                    cos_pixel_values_list[i] = cos_pixel_values

            # Ensure all cos levels have values (fill missing with zeros)
            for i in range(self.cos_level):
                if cos_pixel_values_list[i] is None:
                    if not self.enable_bucket:
                        cos_pixel_values_list[i] = torch.zeros_like(pixel_values)
                    else:
                        cos_pixel_values_list[i] = np.zeros_like(pixel_values)

            # density
            density =  float(data_info['density'])

            return pixel_values, control_pixel_values, subject_image, control_camera_values, text, "video", mask_video_input, depth_pixel_values, cos_pixel_values_list, density, generate_type
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i])
                    width, height = subject_image.size
                    total_pixels = width * height

                    img = padding_image(subject_image, visual_width, visual_height)
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(img)
                subject_image = np.array(subject_images)
            else:
                subject_image = None

            # generate_type for images (default to 'full_tracking')
            generate_type = data_info.get('generate_type', 'full_tracking')
            
            return image, control_image, subject_image, None, text, 'image', None, None, [None] * self.cos_level, None, generate_type
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, subject_image, control_camera_values, name, data_type, mask_video_input, depth_pixel_values, cos_pixel_values_list, density, generate_type = self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["subject_image"] = subject_image
                sample["mask_video_input"] = mask_video_input
                sample["depth_pixel_values"] = depth_pixel_values
                for i, cos_value in enumerate(cos_pixel_values_list):
                    sample[f"cos_pixel_values_{i}"] = cos_value
                sample["text"] = name
                sample["density"] = torch.tensor(1 / density)
                sample["data_type"] = data_type
                sample["generate_type"] = generate_type
                sample["idx"] = idx

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            if generate_type == 'full_tracking':
                mask = get_random_mask(pixel_values.size())
            elif generate_type == 'bg_tracking':
                if mask_video_input is None:
                    raise ValueError(f"mask_video_input is required for generate_type 'bg_tracking'")
                mask = generate_mask_bg_tracking_diable_bucket(mask_video_input)
            elif generate_type == 'fg_tracking':
                if mask_video_input is None:
                    raise ValueError(f"mask_video_input is required for generate_type 'fg_tracking'")
                mask = generate_mask_fg_tracking_diable_bucket(mask_video_input)
            else:
                raise ValueError(f"Unknown generate_type: {generate_type}")
            
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample
