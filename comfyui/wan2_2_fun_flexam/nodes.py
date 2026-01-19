"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import copy
import gc
import json
import os

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar, load_torch_file

from ...FlexAM.data.bucket_sampler import (ASPECT_RATIO_512,
                                               get_closest_ratio)
from ...FlexAM.data.dataset_image_video import process_pose_params
from ...FlexAM.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                                  AutoTokenizer, CLIPModel,
                                  Wan2_2Transformer3DModel, WanT5EncoderModel, Wan2_2Transformer3DModel_FlexAM)
from ...FlexAM.models.cache_utils import get_teacache_coefficients
from ...FlexAM.pipeline import (Wan2_2FunControlPipeline_FlexAM)
from ...FlexAM.ui.controller import all_cheduler_dict
from ...FlexAM.utils.fp8_optimization import (
    convert_model_weight_to_float8, convert_weight_dtype_wrapper,
    replace_parameters_by_name)
from ...FlexAM.utils.lora_utils import merge_lora, unmerge_lora
from ...FlexAM.utils.utils import (filter_kwargs, get_image_latent,
                                       get_image_to_video_latent,
                                       get_video_to_video_latent,
                                       get_maskvideo_to_video_latent,
                                       save_videos_grid)

from ..comfyui_utils import (eas_cache_dir, script_directory,
                             search_model_in_possible_folders, to_pil)
from diffusers import FlowMatchEulerDiscreteScheduler
from ...FlexAM.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from ...FlexAM.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# Used in lora cache
transformer_cpu_cache       = {}
transformer_high_cpu_cache  = {}
# lora path before
lora_path_before            = ""
lora_high_path_before       = ""

def get_wan_scheduler(sampler_name, shift):
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    scheduler_kwargs = {
        "num_train_timesteps": 1000,
        "shift": 5.0,
        "use_dynamic_shifting": False,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
    }
    scheduler_kwargs['shift'] = shift
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, scheduler_kwargs)
    )
    return scheduler

def generate_mask_fg_tracking_for_validation(mask_video_input, blur_radius: int = 15, dilation_pixels: int = 200):
    from scipy.ndimage import gaussian_filter
    import cv2
    """
    从 mask_video_input 生成 refined 二值 mask，处理流程：
    原始mask高斯模糊 → 凸包形状 → 像素扩展 → 二值化
    
    Args:
        mask_video_input: torch.Tensor [F, C, H, W]，数值范围 [0, 1] 或 [0,255]
        blur_radius: 高斯模糊半径
        dilation_pixels: 像素扩展数量
    
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

def generate_mask_bg_tracking_for_validation(mask_video_input):
    """
    从 mask_video_input 生成背景跟踪的 mask
    
    Args:
        mask_video_input: torch.Tensor [F, C, H, W]
    
    Returns:
        mask: torch.Tensor [F, 1, H, W], dtype=torch.uint8, 值为 {0,1}
              - 背景区域为 1，前景区域为 0
              - 第 0 帧始终全 0
    """
    f, c, h, w = mask_video_input.shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    # 转灰度
    if c > 1:
        gray_frames = mask_video_input.mean(dim=1, keepdim=True)  # [F,1,H,W]
    else:
        gray_frames = mask_video_input

    for i in range(1, f):  # 跳过第 0 帧
        frame = gray_frames[i,0].cpu().numpy()  # [H,W]
        
        # 二值化并反转（背景为1，前景为0）
        bg_mask = (frame <= 0.5).astype(np.uint8)
        mask[i,0] = torch.from_numpy(bg_mask)

    # 确保第 0 帧全 0
    mask[0,0] = 0

    return mask


class LoadWan2_2FunModel_FlexAM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [ 
                        'Wan2.2-Fun-A14B-InP',
                        'Wan2.2-Fun-A14B-Control',
                        'Wan2.2-Fun-A14B-Control-Camera',
                        'Wan2.2-Fun-5B-InP',
                        'Wan2.2-Fun-5B-Control',
                        'Wan2.2-Fun-5B-Control-Camera',
                    ],
                    {
                        "default": 'Wan2.2-Fun-5B-Control',
                    }
                ),
                "model_type": (
                    ["Inpaint", "Control"],
                    {
                        "default": "Control",
                    }
                ),
                "GPU_memory_mode":(
                    ["model_full_load", "model_full_load_and_qfloat8","model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                    {
                        "default": "model_cpu_offload",
                    }
                ),
                "config": (
                    [
                        "wan2.2/wan_civitai_i2v.yaml",
                        "wan2.2/wan_civitai_5b.yaml",
                        "wan2.2/wan_civitai_5b_FlexAM.yaml",
                    ],
                    {
                        "default": "wan2.2/wan_civitai_5b_FlexAM.yaml",
                    }
                ),
                "precision": (
                    ['fp16', 'bf16'],
                    {
                        "default": 'fp16'
                    }
                ),
            },
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, GPU_memory_mode, model_type, model, precision, config):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Init processbar
        pbar = ProgressBar(5)

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)

        # Detect model is existing or not
        possible_folders = ["CogFlexAM", "Fun_Models", "FlexAM", "Wan-AI"] + \
                [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        # Initialize model_name as None
        model_name = search_model_in_possible_folders(possible_folders, model)

        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        # Update pbar
        pbar.update(1)
        
        # Get Transformer
        transformer = Wan2_2Transformer3DModel_FlexAM.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype,
        )

        if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
            transformer_2 = Wan2_2Transformer3DModel_FlexAM.from_pretrained(
                os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                low_cpu_mem_usage=False,
                torch_dtype=weight_dtype,
            )
        else:
            transformer_2 = None
        # Update pbar
        pbar.update(1) 

        # Get tokenizer and text_encoder
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        pbar.update(1) 

        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype,
        )
        pbar.update(1) 

        # Get pipeline
        if model_type == "Inpaint":
            if transformer.config.in_channels != vae.config.latent_channels:
                pipeline = Wan2_2FunInpaintPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    transformer_2=transformer_2,
                    scheduler=scheduler,
                )
            else:
                pipeline = Wan2_2FunPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    transformer_2=transformer_2,
                    scheduler=scheduler,
                )
        else:
            pipeline = Wan2_2FunControlPipeline_FlexAM(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                transformer_2=transformer_2,
                scheduler=scheduler,
            )

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation",], device=device)
                transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        funmodels = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
            'config': config,
        }
        return (funmodels,)

class Wan2_2FunV2VSampler_FlexAM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": (
                    "FunModels", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 161, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        640,
                        768,
                        896,
                        960,
                        1024,
                    ], {"default": 640}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "denoise_strength": (
                    "FLOAT", {"default": 1.00, "min": 0.05, "max": 1.00, "step": 0.01}
                ),
                "scheduler": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 5, "min": 1, "max": 100, "step": 1}
                ),
                "boundary": (
                    "FLOAT", {"default": 0.900, "min": 0.00, "max": 1.00, "step": 0.001}
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}
                ),
                "enable_teacache":(
                    [False, True],  {"default": True,}
                ),
                "num_skip_start_steps": (
                    "INT", {"default": 5, "min": 0, "max": 50, "step": 1}
                ),
                "teacache_offload":(
                    [False, True],  {"default": True,}
                ),
                "cfg_skip_ratio":(
                    "FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}
                ),
                "generate_type": (
                    ["motion_transfer", "fg_generation", "bg_generation"],
                    {"default": "motion_transfer"}
                ),
                "dilation_pixels": (
                    "INT", {"default": 200, "min": 0, "max": 1000, "step": 10}
                ),
            },
            "optional": {
                "original_video": ("IMAGE",),
                "depth_video": ("IMAGE",),
                "control_video": ("IMAGE",),
                "cos_video0": ("IMAGE",),
                "cos_video1": ("IMAGE",),
                "cos_video2": ("IMAGE",),
                "cos_video3": ("IMAGE",),
                "mask_video": ("IMAGE",),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "ref_image": ("IMAGE",),
                "camera_conditions": ("STRING", {"forceInput": True}),
                "riflex_k": ("RIFLEXT_ARGS",),
                "density": (
                    "INT", {"default": 10, "min": 1}
                ),                
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, denoise_strength, scheduler, shift, boundary, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, generate_type, dilation_pixels, original_video=None, control_video=None, depth_video=None, cos_video0=None, cos_video1=None, cos_video2=None, cos_video3=None, mask_video=None, start_image=None, end_image=None, ref_image=None, camera_conditions=None, riflex_k=0, density=10):
        global transformer_cpu_cache
        global transformer_high_cpu_cache
        global lora_path_before
        global lora_high_path_before

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        weight_dtype = funmodels['dtype']
        model_type = funmodels['model_type']

        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        if original_video is not None:
            if isinstance(original_video, torch.Tensor):
                original_video_np = np.array(original_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(original_video_np[0]).size
            else:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(original_video).read()[1]).size
        elif control_video is not None:
            if isinstance(control_video, torch.Tensor):
                control_video_np = np.array(control_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(control_video_np[0]).size
            else:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
        else:
            original_width, original_height = 384 / 512 * base_resolution, 672 / 512 * base_resolution

        print("original_width,original_height",original_width,original_height)

        # Process other inputs
        if ref_image is not None:
            ref_image = [to_pil(_ref_image) for _ref_image in ref_image]
            
        if start_image is not None:
            start_image = [to_pil(_start_image) for _start_image in start_image]
            
        if end_image is not None:
            end_image = [to_pil(_end_image) for _end_image in end_image]

        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]

        # Load Sampler
        pipeline.scheduler = get_wan_scheduler(scheduler, shift)
        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        else:
            pipeline.transformer.disable_teacache()
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
                if pipeline.transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

            if generate_type == "motion_transfer":
                inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(start_image, end_image, video_length=video_length, sample_size=(height, width))
            else:
                # fg_generation 或 bg_generation 需要mask_video
                if mask_video is None:
                    raise ValueError(f"For {generate_type}, mask_video is required")
                    
                if start_image is None:
                    raise ValueError(f"For {generate_type}, start_image (repainted first frame) is required")
                if isinstance(mask_video, str):
                    mask_video_input = get_maskvideo_to_video_latent(mask_video, video_length=video_length, sample_size=(height, width))
                else:
                    if isinstance(mask_video, torch.Tensor):
                        mask_video = mask_video.float()
                    else:
                        mask_video = torch.from_numpy(np.array(mask_video)).float()
                    
                    from torchvision.transforms.functional import resize
                    mask_video_input = torch.stack([resize(mask_frame, (height, width)) for mask_frame in mask_video[:video_length].permute(0, 3, 1, 2).contiguous()], dim=0)
                if original_video is None:
                    raise ValueError(f"For {generate_type}, original_video is required")
                
                original_video = original_video * 255
                original_video_latent = get_video_to_video_latent(original_video, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)[0]
                    
                start_image_latent = get_image_latent(start_image[0], sample_size=(height, width))
                
                inpaint_video = torch.cat([start_image_latent[:, :, :1], original_video_latent[:, :, 1:]], dim=2)
                
                if generate_type == "fg_generation":
                    inpaint_video_mask = generate_mask_fg_tracking_for_validation(mask_video_input, dilation_pixels=dilation_pixels)
                else:  # bg_generation
                    inpaint_video_mask = generate_mask_bg_tracking_for_validation(mask_video_input)
                
                # 转换mask格式为pipeline需要的格式
                inpaint_video_mask = (inpaint_video_mask * 255).unsqueeze(0).permute(0, 2, 1, 3, 4)
                
            if ref_image is not None:
                ref_image = get_image_latent(ref_image[0] if ref_image is not None else ref_image, sample_size=(height, width))
                
            if camera_conditions is not None and len(camera_conditions) > 0: 
                poses      = json.loads(camera_conditions)
                cam_params = np.array([[float(x) for x in pose] for pose in poses])
                cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
                control_camera_video = process_pose_params(cam_params, width=width, height=height)
                control_camera_video = control_camera_video[:video_length].permute([3, 0, 1, 2]).unsqueeze(0)
                input_video, input_video_mask = None, None
            else:
                control_camera_video = None
                input_video, input_video_mask, _, _ = get_video_to_video_latent(control_video, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)
                depth_video, depth_video_mask, _, _ = get_video_to_video_latent(depth_video, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)
                cos_video0, cos_video0_mask, _, _ = get_video_to_video_latent(cos_video0, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)
                cos_video1, cos_video1_mask, _, _ = get_video_to_video_latent(cos_video1, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)
                cos_video2, cos_video2_mask, _, _ = get_video_to_video_latent(cos_video2, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)
                cos_video3, cos_video3_mask, _, _ = get_video_to_video_latent(cos_video3, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)
                cos_video_dict = {0: cos_video0, 1: cos_video1, 2: cos_video2, 3: cos_video3}
         

            # Apply lora
            if funmodels.get("lora_cache", False):
                if len(funmodels.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(funmodels.get("loras", []) + funmodels.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
                   
                    if pipeline.transformer_2 is not None:
                        # Save the original weights to cpu
                        if len(transformer_high_cpu_cache) == 0:
                            print('Save transformer high state_dict to cpu memory')
                            transformer_high_state_dict = pipeline.transformer_2.state_dict()
                            for key in transformer_high_state_dict:
                                transformer_high_cpu_cache[key] = transformer_high_state_dict[key].clone().cpu()

                        lora_high_path_now = str(funmodels.get("loras_high", []) + funmodels.get("strength_model", []))
                        if lora_high_path_now != lora_high_path_before:
                            print('Merge Lora High with Cache')
                            lora_high_path_before = copy.deepcopy(lora_high_path_now)
                            pipeline.transformer_2.load_state_dict(transformer_cpu_cache)
                            for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                                pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
            else:
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if pipeline.transformer_2 is not None:
                    if len(transformer_high_cpu_cache) != 0:
                        pipeline.transformer_2.load_state_dict(transformer_high_cpu_cache)
                        transformer_high_cpu_cache = {}
                        lora_high_path_before = ""
                        gc.collect()

                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
            print("inpaint_video.shape",inpaint_video.shape)
            print("inpaint_video_mask.shape",inpaint_video_mask.shape)
            print("input_video.shape",input_video.shape)
            print("depth_video.shape",depth_video.shape)
            print("ref_image.shape",ref_image.shape)
            # Always use control mode
            sample = pipeline(
                    prompt, 
                    num_frames = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,

                    video      = inpaint_video,
                    mask_video   = inpaint_video_mask,
                    control_video = input_video,
                    control_camera_video = control_camera_video,
                    depth_video = depth_video,
                    cos_control_videos = cos_video_dict,
                    cos_level= 4,
                    density = 1.0 / max(density, 1),
                    ref_image = ref_image,
                    boundary = boundary,
                    comfyui_progressbar = True,
                ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
                if pipeline.transformer_2 is not None:
                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
        return (videos,)   
