import gc
import inspect
import os
import shutil
import subprocess
import time

import cv2
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            # write video
            writer = imageio.get_writer(
                path, fps=fps, codec='libx264', quality=8)
            for frame in outputs:
                writer.append_data(np.array(frame))
            writer.close()
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)

def save_videos_comparison(tracking_path, original_path, generated_path, ref_image_path, output_path, fps=16):

    from moviepy.editor import ImageSequenceClip
    
    """Create a 2x2 comparison video of four videos/images:
       Top left: Generated video, Top right: Original video (GT)
       Bottom left: Reference image/video, Bottom right: Tracking video
       Saved using MoviePy.
    """
    def load_video_frames(gt_video_path, target_size_wh): # target_size_wh is (width, height)
        cap_load = cv2.VideoCapture(gt_video_path)
        if not cap_load.isOpened():
            print(f"Warning: Unable to open video {gt_video_path}. Returning empty frame list.")
            return []
        frames = []
        while True:
            ret, frame = cap_load.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size_wh)
            frames.append(frame)
        cap_load.release()
        return frames
    
    cap_gen_check = cv2.VideoCapture(generated_path)
    if not cap_gen_check.isOpened():
        print(f"Error: Unable to open generated video {generated_path} to get dimensions.")
        # Try to get dimensions from another video
        cap_alt_check = cv2.VideoCapture(original_path)
        if not cap_alt_check.isOpened():
            print(f"Error: Unable to open original video {original_path}. Cannot determine base dimensions.")
            return
        base_width = int(cap_alt_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        base_height = int(cap_alt_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_alt_check.release()
    else:
        base_width = int(cap_gen_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        base_height = int(cap_gen_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_gen_check.release()

    target_dim_wh = (base_width, base_height)

    generated_frames = load_video_frames(generated_path, target_dim_wh)
    original_frames = load_video_frames(original_path, target_dim_wh)
    tracking_frames = load_video_frames(tracking_path, target_dim_wh)

    # Handle reference: could be image or video
    ref_frames = []
    if ref_image_path and os.path.exists(ref_image_path):
        # Check if it's a video file
        if ref_image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv')):
            # It's a video, load all frames
            ref_frames = load_video_frames(ref_image_path, target_dim_wh)
        else:
            # It's an image, load once and reuse
            try:
                # Use PIL Image instead of cv2.imread to handle corrupted images better
                pil_img = Image.open(ref_image_path).convert('RGB')
                # Convert PIL image to BGR format for OpenCV
                ref_img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                ref_frame_resized = cv2.resize(ref_img_bgr, target_dim_wh)
                ref_frames = [ref_frame_resized]  # Single frame in list
            except Exception as e:
                print(f"Error loading reference image {ref_image_path}: {e}")
                ref_frames = []
    
    # If ref_frames is empty, create a black frame
    if not ref_frames:
        ref_frames = [np.zeros((base_height, base_width, 3), dtype=np.uint8)]
        print("Warning: No reference frames found. Using black frame.")

    if not generated_frames or not original_frames or not tracking_frames:
        print("Error: One or more videos failed to load frames. Cannot create comparison video.")
        return
        
    min_length = min(len(generated_frames), len(original_frames), len(tracking_frames))
    if min_length == 0:
        print("Error: Video frame counts are inconsistent or one video is empty, cannot create comparison video.")
        return

    generated_frames = generated_frames[:min_length]
    original_frames = original_frames[:min_length]
    tracking_frames = tracking_frames[:min_length]
    
    combined_frames_for_moviepy = []
    
    for i in range(min_length):
        gen_f = generated_frames[i]
        orig_f = original_frames[i]
        track_f = tracking_frames[i]

        # Get reference frame: if single frame, reuse it; if multiple frames, use corresponding frame
        if len(ref_frames) == 1:
            ref_frame = ref_frames[0]  # Single frame, reuse for all
        else:
            # Multiple frames, use corresponding frame (cycle if ref is shorter)
            ref_frame = ref_frames[i % len(ref_frames)]

        # Create top row: Generated | Original
        top_row = np.hstack([gen_f, orig_f])
        
        # Create bottom row: Reference Image/Video | Tracking
        bottom_row = np.hstack([ref_frame, track_f])
        
        # Vertically stack top and bottom rows
        combined_frame = np.vstack([top_row, bottom_row])
        # Final combined_frame shape: (2 * base_height, 2 * base_width, 3)
        
        # Add labels (only on the first frame)
        if i == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 # Adjust font size
            font_color = (255, 255, 255)  # White
            font_thickness = 2
            alpha = 0.6 # Background transparency

            labels_info = [
                {"text": "Generated", "pos": (10, 30)}, # Top-left
                {"text": "Original",  "pos": (base_width + 10, 30)}, # Top-right
                {"text": "Reference", "pos": (10, base_height + 30)}, # Bottom-left
                {"text": "Tracking",  "pos": (base_width + 10, base_height + 30)}  # Bottom-right
            ]
            
            for label_info in labels_info:
                label_text = label_info["text"]
                text_x, text_y = label_info["pos"]
                
                text_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                
                rect_x1 = max(0, text_x - 5)
                rect_y1 = max(0, text_y - text_h - 5) # y is baseline, so text_h is above it
                rect_x2 = min(combined_frame.shape[1], text_x + text_w + 5)
                rect_y2 = min(combined_frame.shape[0], text_y + 5)

                if rect_x1 < rect_x2 and rect_y1 < rect_y2:
                    overlay = combined_frame.copy()
                    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, alpha, combined_frame, 1 - alpha, 0, combined_frame)
                
                cv2.putText(combined_frame, label_text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        combined_frames_for_moviepy.append(combined_frame_rgb)
    
    if not combined_frames_for_moviepy:
        print("No frames available to create video.")
        return

    clip = ImageSequenceClip(combined_frames_for_moviepy, fps=fps)
    clip.write_videofile(output_path, codec='libx264', fps=fps, logger=None)


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        print("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"merge_video_audio failed with error: {e}")

def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path
            if isinstance(input_video, torch.Tensor):
                import torch.nn.functional as F
                input_video = F.interpolate(
                    input_video.permute(0, 3, 1, 2),
                    size=(sample_size[0], sample_size[1]),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                input_video = input_video.cpu().numpy()

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image


def get_maskvideo_to_video_latent(mask_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    import decord
    import numpy as np
    from torchvision.transforms.functional import resize
    mask_frames = None
    if mask_path:
        mask_reader = decord.VideoReader(uri=str(mask_path))
        mask_num_frames = len(mask_reader)
        
        if mask_num_frames < video_length:
            # Read all available frames
            mask_frame_indices = list(range(mask_num_frames))
            mask_frames = mask_reader.get_batch(mask_frame_indices)
            
            # Handle both numpy arrays and tensors from decord
            if isinstance(mask_frames, np.ndarray):
                mask_frames = torch.from_numpy(mask_frames).float()
            elif hasattr(mask_frames, 'asnumpy'):  # decord.ndarray.NDArray
                mask_frames = torch.from_numpy(mask_frames.asnumpy()).float()
            else:
                mask_frames = mask_frames.float()
            mask_frames = mask_frames.permute(0, 3, 1, 2).contiguous()
            # Resize frames first
            mask_frames_resized = torch.stack([resize(mask_frame, (sample_size[0], sample_size[1])) for mask_frame in mask_frames], dim=0)
            # Pad by repeating the last frame
            last_frame = mask_frames_resized[-1:].clone()  # Get last frame and keep batch dimension
            padding_frames = last_frame.repeat(video_length - mask_num_frames, 1, 1, 1)
            mask_frames = torch.cat([mask_frames_resized, padding_frames], dim=0)
        else:
            # Read only video_length frames when mask has enough frames
            mask_frame_indices = list(range(video_length))
            mask_frames = mask_reader.get_batch(mask_frame_indices)
            
            # Handle both numpy arrays and tensors from decord
            if isinstance(mask_frames, np.ndarray):
                mask_frames = torch.from_numpy(mask_frames).float()
            elif hasattr(mask_frames, 'asnumpy'):  # decord.ndarray.NDArray
                mask_frames = torch.from_numpy(mask_frames.asnumpy()).float()                
            else:
                mask_frames = mask_frames.float()
            mask_frames = mask_frames[:video_length]
            mask_frames = mask_frames.permute(0, 3, 1, 2).contiguous()
            mask_frames = torch.stack([resize(mask_frame, (sample_size[0], sample_size[1])) for mask_frame in mask_frames], dim=0)

    return mask_frames


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

def get_image_latent(ref_image=None, sample_size=None, padding=False):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        elif isinstance(ref_image, Image.Image):
            ref_image = ref_image.convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

    return ref_image

def timer(func):
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        end_time    = time.time()
        print(f"function {func.__name__} running for {end_time - start_time} seconds")
        return result
    return wrapper

def timer_record(model_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.time()
            result      = func(*args, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    time_sum  = end_time - start_time
                    print('# --------------------------------------------------------- #')
                    print(f'#   {model_name} time: {time_sum}s')
                    print('# --------------------------------------------------------- #')
                    _write_to_excel(model_name, time_sum)
            else:
                time_sum  = end_time - start_time
                print('# --------------------------------------------------------- #')
                print(f'#   {model_name} time: {time_sum}s')
                print('# --------------------------------------------------------- #')
                _write_to_excel(model_name, time_sum)
            return result
        return wrapper
    return decorator

def _write_to_excel(model_name, time_sum):
    import os

    import pandas as pd

    row_env = os.environ.get(f"{model_name}_EXCEL_ROW", "1")  # 默认第1行
    col_env = os.environ.get(f"{model_name}_EXCEL_COL", "1")  # 默认第A列
    file_path = os.environ.get("EXCEL_FILE", "timing_records.xlsx")  # 默认文件名

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    row_idx = int(row_env)
    col_idx = int(col_env)

    if row_idx >= len(df):
        df = pd.concat([df, pd.DataFrame([ [None] * (len(df.columns) if not df.empty else 0) ] * (row_idx - len(df) + 1))], ignore_index=True)

    if col_idx >= len(df.columns):
        df = pd.concat([df, pd.DataFrame(columns=range(len(df.columns), col_idx + 1))], axis=1)

    df.iloc[row_idx, col_idx] = time_sum

    df.to_excel(file_path, index=False, header=False, sheet_name="Sheet1")
