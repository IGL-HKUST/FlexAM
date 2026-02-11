import os
import sys

# Set up paths for all submodules before any other imports
project_root = os.path.dirname(os.path.abspath(__file__))
for path in ["submodules/MoGe", "submodules/vggt", "submodules/DELTA", 
             "submodules/DELTA/densetrack3d", "submodules/Pi3"]:
    sys.path.insert(0, os.path.join(project_root, path))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import torchvision.transforms.functional as TF
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from diffusers.utils import load_image, load_video

from pipelines import FlexAMPipeline, FirstFrameRepainter, CameraMotionGenerator, ObjectMotionGenerator
from submodules.MoGe.moge.model.v2 import MoGeModel
from submodules.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from submodules.vggt.vggt.models.vggt import VGGT
from FlexAM.utils.utils import (filter_kwargs,
                                    get_image_to_video_latent, 
                                    get_image_latent,
                                    get_video_to_video_latent,
                                    get_maskvideo_to_video_latent,
                                    get_image_to_video_latent,
                                    save_videos_grid)

def generate_mask_fg_tracking_for_validation(mask_video_input, blur_radius: int = 15, dilation_pixels: int = 200):
    from scipy.ndimage import gaussian_filter
    import cv2
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
    # dilation_pixels 现在作为参数传入
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

def _is_video_file(file_path) -> bool:
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
    # Convert to Path object if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.suffix.lower() in video_extensions

def _is_image_file(file_path) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    # Convert to Path object if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.suffix.lower() in image_extensions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Path to input video/image')
    parser.add_argument('--prompt', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM", help='Path to model checkpoint')
    parser.add_argument('--num_inference_steps', type=int, default=40, help='Number of inference steps')
    parser.add_argument('--repaint', type=str, default=None, 
                       help='Path to repainted image, or "true" to perform repainting, if not provided use original frame')
    parser.add_argument('--camera_motion', type=str, default=None, 
                    help='Camera motion mode: "trans <dx> <dy> <dz>" or "rot <axis> <angle>" or "spiral <radius>" or "path"')
    parser.add_argument('--pose_file', type=str, default=None,
                    help='Path to pose file (txt or mp4) for camera motion. Txt file contains camera parameters directly, mp4 file will use Pi3 to estimate camera parameters')
    parser.add_argument('--override_extrinsics', type=str, default="append", choices=["override", "append"],
                help='How to apply camera motion: "override" to replace original camera, "append" to build upon it. Override is experimental and may not work as expected.')
    parser.add_argument('--object_motion', type=str, default=None, help='Object motion mode: up/down/left/right')
    parser.add_argument('--object_mask', type=str, default=None, help='Path to object mask image (binary image)')
    parser.add_argument('--tracking_method', type=str, default='DELTA', choices=['DELTA', 'moge'], help='Tracking method to use (DELTA or moge)')
    parser.add_argument('--sample_size', type=int, nargs=2, default=[512, 896], help='Sample size [height, width]')
    parser.add_argument('--video_length', type=int, default=49, help='Video length in frames')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask video, mask video and ref video should provided together')
    parser.add_argument('--generate_type', type=str, default='full_edit', help='generate type: choose from full_edit, foreground_edit, background_edit')
    parser.add_argument('--density', type=int, default=10, help='Density level of tracking points, higher values result in sparser tracking points')
    parser.add_argument('--dilation_pixels', type=int, default=200, help='Dilation pixels for mask processing in foreground_edit mode')

    args = parser.parse_args()

    cos_level = 4
    fps = 16

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Load input video/image
    if not _is_video_file(args.input_path) and not _is_image_file(args.input_path):
        raise ValueError(f"Input file '{args.input_path}' is not a valid video or image file. Supported video formats: .mp4, .avi, .mov, .mkv, .webm, .m4v, .flv, .wmv. Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
    
    is_video = _is_video_file(args.input_path)
    if is_video:
        input_video_tensor, _, _, _ = get_video_to_video_latent(args.input_path, video_length=args.video_length, sample_size=args.sample_size, fps=None, ref_image=None)
    else:
        input_video_tensor = get_image_latent(args.input_path, sample_size=args.sample_size)

    if not is_video:
        args.tracking_method = "moge"
        print("Image input detected, using MoGe for tracking video generation.")

    # Initialize pipeline
    pipe = FlexAMPipeline(gpu_id=args.gpu, output_dir=args.output_dir)
    pipe.fps = fps
    if args.tracking_method == "moge" and args.tracking_path is None:
        moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(pipe.device)
    
    # Repaint first frame if requested
    repaint_img_tensor = None
    if args.repaint:
        if args.repaint.lower() == "true":
            repainter = FirstFrameRepainter(height = args.sample_size[0], width = args.sample_size[1], gpu_id=args.gpu, output_dir=args.output_dir)
            repaint_img_tensor = repainter.repaint(
                input_video_tensor[0,:,0], 
                prompt=args.prompt,
                depth_path=None,
            )
            repaint_img_tensor=repaint_img_tensor.unsqueeze(0).unsqueeze(2)
            
        else:
            repaint_img_tensor =  get_image_latent(args.repaint, sample_size=args.sample_size)

    # Generate tracking if not provided
    tracking_tensor = None
    pred_tracks = None
    cam_motion = CameraMotionGenerator(args.camera_motion, frame_num=args.video_length, H=args.sample_size[0], W=args.sample_size[1], pose_file=args.pose_file)

    if args.tracking_method == "moge":
        # Use the first frame from previously loaded input_video_tensor
        infer_result = moge.infer((input_video_tensor[0,:,0]).to(pipe.device))  # [C, H, W] in range [0,1]
        H, W = infer_result["points"].shape[0:2]
        pred_tracks = infer_result["points"].unsqueeze(0).repeat(args.video_length, 1, 1, 1) #[T, H, W, 3]
        cam_motion.set_intr(infer_result["intrinsics"])

        # Apply object motion if specified
        if args.object_motion:
            if args.object_mask is None:
                raise ValueError("Object motion specified but no mask provided. Please provide a mask image with --object_mask")
                
            # Load mask image
            mask_image = Image.open(args.object_mask).convert('L')  # Convert to grayscale
            mask_image = transforms.Resize((args.sample_size[0], args.sample_size[1]))(mask_image)  # Resize to match video size
            # Convert to binary mask
            mask = torch.from_numpy(np.array(mask_image) > 127)  # Threshold at 127
            
            motion_generator = ObjectMotionGenerator(device=pipe.device)

            pred_tracks = motion_generator.apply_motion(
                pred_tracks=pred_tracks,
                mask=mask,
                motion_type=args.object_motion,
                distance=50,
                num_frames=args.video_length,
                tracking_method="moge"
            )
            print("Object motion applied")

        # Apply camera motion if specified
        if args.camera_motion:
            poses = cam_motion.get_default_motion() # shape: [T, 4, 4]
            print("Camera motion applied")
        else:
            # no poses
            poses = torch.eye(4).unsqueeze(0).repeat(args.video_length, 1, 1)
        # change pred_tracks into screen coordinate
        pred_tracks_flatten = pred_tracks.reshape(args.video_length, H*W, 3)
        pred_tracks = cam_motion.w2s_moge(pred_tracks_flatten, poses) # [T, H * W, 3]

        # Convert moge format to delta format for visualization
        pred_tracks_moge_format = pred_tracks.reshape(args.video_length, H, W, 3)  # [T, H, W, 3]
        delta_points, vis_mask = pipe.convert_moge_to_delta_format(
            pred_tracks_moge_format.cpu().numpy(), 
            infer_result["mask"].cpu().numpy(), 
            height=args.sample_size[0], 
            width=args.sample_size[1]
        )

        tracking_tensor, cos_video_dict, depth_video = pipe.visualize_tracking_DELTA(
            delta_points, vis_mask=vis_mask, point_wise=2, 
            height=args.sample_size[0], width=args.sample_size[1], cos_level=cos_level,
            generate_type=args.generate_type, mask_path=args.mask_path
        )


        if repaint_img_tensor is not None:
            full_ref=repaint_img_tensor
            first_frame=repaint_img_tensor[0,:,0]
            clip_image_pil = TF.to_pil_image((first_frame*255).byte())
        else:
            first_frame = input_video_tensor[0,:,0] # [B, C, T, H, W]  -> # c h w
            clip_image_pil = TF.to_pil_image((first_frame*255).byte())
            full_ref = get_image_latent(clip_image_pil, sample_size=args.sample_size)
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "clip_image.png")
        clip_image_pil.save(temp_image_path)
        inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(temp_image_path, None, video_length=args.video_length, sample_size=args.sample_size)


        print('export tracking video via MoGe.')

    else:

        pred_tracks, pred_visibility = pipe.generate_tracking_DELTA(input_video_tensor, density=args.density) # T N 3, T N

        # Preprocess video tensor to match VGGT requirements
        b, c, t, h, w = input_video_tensor.shape #[B, C, T, H, W]
        new_width = 518
        new_height = round(h * (new_width / w) / 14) * 14
        resize_transform = transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC)
        # Convert [B, C, T, H, W] -> [T, C, H, W] for VGGT
        video_vggt = input_video_tensor.squeeze(0).permute(1, 0, 2, 3)  # [T, C, H, W]
        video_vggt = resize_transform(video_vggt)
        
        if new_height > 518:
            start_y = (new_height - 518) // 2
            video_vggt = video_vggt[:, :, start_y:start_y + 518, :]

        # Get extrinsic and intrinsic matrices
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(pipe.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=pipe.dtype):

                video_vggt = video_vggt.unsqueeze(0)  # [1, T, C, H, W]
                aggregated_tokens_list, ps_idx = vggt_model.aggregator(video_vggt.to(pipe.device))
            
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extr, intr = pose_encoding_to_extri_intri(vggt_model.camera_head(aggregated_tokens_list)[-1], video_vggt.shape[-2:])
                depth_map, depth_conf = vggt_model.depth_head(aggregated_tokens_list, video_vggt, ps_idx)
        
        cam_motion.set_intr(intr)
        cam_motion.set_extr(extr)

        del vggt_model

        # Apply camera motion if specified
        if args.camera_motion:
            poses = cam_motion.get_default_motion() # shape: [T, 4, 4]
            pred_tracks_world = cam_motion.s2w_vggt(pred_tracks, extr, intr)
            pred_tracks = cam_motion.w2s_vggt(pred_tracks_world, extr, intr, poses, 
                                 override_extrinsics=(args.override_extrinsics == "override"))
            print("Camera motion applied")
        
        # Apply object motion if specified
        if args.object_motion:
            if args.object_mask is None:
                raise ValueError("Object motion specified but no mask provided. Please provide a mask image with --object_mask")
                
            # Load mask image
            mask_image = Image.open(args.object_mask).convert('L')  # Convert to grayscale
            mask_image = transforms.Resize((args.sample_size[0], args.sample_size[1]))(mask_image)  # Resize to match video size
            # Convert to binary mask
            mask = torch.from_numpy(np.array(mask_image) > 127)  # Threshold at 127
            
            motion_generator = ObjectMotionGenerator(device=pipe.device)
            
            pred_tracks = motion_generator.apply_motion(
                pred_tracks=pred_tracks,
                mask=mask,
                motion_type=args.object_motion,
                distance=50,
                num_frames=args.video_length,
                tracking_method="DELTA"
            )
            print(f"Object motion '{args.object_motion}' applied using mask from {args.object_mask}")

        tracking_tensor, cos_video_dict, depth_video = pipe.visualize_tracking_DELTA(pred_tracks, pred_visibility,  height=args.sample_size[0], width=args.sample_size[1] ,cos_level=cos_level, generate_type=args.generate_type, mask_path=args.mask_path)

        # clip_image_pil
        if args.generate_type == 'full_edit':
            if repaint_img_tensor is not None:
                full_ref=repaint_img_tensor
                first_frame=repaint_img_tensor[0,:,0]
                clip_image_pil = TF.to_pil_image((first_frame * 255).byte())
            else:
                first_frame = input_video_tensor[0,:,0] # [B, C, T, H, W]  -> # c h w
                clip_image_pil = TF.to_pil_image((first_frame * 255).byte())
                full_ref = get_image_latent(clip_image_pil, sample_size=args.sample_size)

            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_image_path = os.path.join(temp_dir, "clip_image.png")
            clip_image_pil.save(temp_image_path)
            inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(temp_image_path, None, video_length=args.video_length, sample_size=args.sample_size)
        else:
            if repaint_img_tensor is None:
                raise ValueError("repaint must be provided for foreground_edit/background_edit")
            
            mask_video_input = get_maskvideo_to_video_latent(
                args.mask_path, 
                video_length=args.video_length, 
                sample_size=args.sample_size
            )
            if args.generate_type == 'foreground_edit':
                inpaint_video_mask = generate_mask_fg_tracking_for_validation(mask_video_input, dilation_pixels=args.dilation_pixels)
            elif args.generate_type == 'background_edit':
                inpaint_video_mask = generate_mask_bg_tracking_for_validation(mask_video_input)
            inpaint_video_mask = (inpaint_video_mask * 255).unsqueeze(0).permute(0, 2, 1, 3, 4)
            
            # Compose inpaint_video: repaint_img_tensor as first frame + input_video_tensor subsequent frames
            inpaint_video = torch.cat([repaint_img_tensor[:, :, :1], input_video_tensor[:, :, 1:]], dim=2)
            first_frame = repaint_img_tensor[0,:,0]  # c h w
            clip_image_pil = TF.to_pil_image((first_frame * 255).byte())
            full_ref = get_image_latent(clip_image_pil, sample_size=args.sample_size)
    

    pipe.apply_tracking(
        fps=fps,
        tracking_tensor=tracking_tensor,
        cos_video_dict = cos_video_dict,
        depth_video = depth_video,
        cos_level = cos_level,
        full_ref = full_ref,
        inpaint_video      = inpaint_video,
        inpaint_video_mask   = inpaint_video_mask,        
        prompt=args.prompt,
        checkpoint_path=args.checkpoint_path,
        num_inference_steps=args.num_inference_steps,
        height = args.sample_size[0],
        width = args.sample_size[1],
        video_length = args.video_length,
        density = args.density,
        seed = 1245644,
    )
