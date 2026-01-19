import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from PIL import Image


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


def generate_mask(mask_video_input):
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
    if mask_video_input is None:
        return None
    
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
        binary_mask = (normalized > 0.5).float()
        
        mask[frame_idx] = binary_mask
    
    return mask


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

class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds

class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []

###  video based dataset with tracking, ref, depth and explicit density
class VideoDatasetMultiontrol(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.enable_inpaint = kwargs.pop("enable_inpaint", True)
        self.ref_column = kwargs.pop("ref_column", "ref")
        self.mask_column = kwargs.pop("mask_column", "mask")
        self.depth_column = kwargs.pop("depth_column", None)
        self.cos_column = kwargs.pop("cos_column", None)
        self.cos_level = kwargs.pop("cos_level", 4)
        self.density_column = kwargs.pop("density_column", "density")

        data_root = args[0] if args else kwargs.get('data_root')
        dataset_file = args[1] if len(args) > 1 else kwargs.get('dataset_file')
        caption_column = args[2] if len(args) > 2 else kwargs.get('caption_column', 'text')
        video_column = args[3] if len(args) > 3 else kwargs.get('video_column', 'video')
        max_num_frames = args[4] if len(args) > 4 else kwargs.get('max_num_frames', 49)
        id_token = args[5] if len(args) > 5 else kwargs.get('id_token')
        height_buckets = args[6] if len(args) > 6 else kwargs.get('height_buckets')
        width_buckets = args[7] if len(args) > 7 else kwargs.get('width_buckets')
        frame_buckets = args[8] if len(args) > 8 else kwargs.get('frame_buckets')
        load_tensors = args[9] if len(args) > 9 else kwargs.get('load_tensors', False)
        random_flip = args[10] if len(args) > 10 else kwargs.get('random_flip')
        image_to_video = args[11] if len(args) > 11 else kwargs.get('image_to_video', False)
        

        
        Dataset.__init__(self)
        
        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def _is_video_file(self, file_path) -> bool:
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
        # Convert to Path object if it's a string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return file_path.suffix.lower() in video_extensions

    def _is_image_file(self, file_path) -> bool:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        # Convert to Path object if it's a string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return file_path.suffix.lower() in image_extensions

    def _preprocess_video(self, path: Path, tracking_path: Path, ref_path: Optional[Path] = None, mask_path: Optional[Path] = None, depth_path: Optional[Path] = None, cos_0_path: Optional[Path] = None) -> torch.Tensor:
        # Ensure all paths are Path objects
        if isinstance(path, str):
            path = Path(path)
        if isinstance(tracking_path, str):
            tracking_path = Path(tracking_path)
        if isinstance(ref_path, str):
            ref_path = Path(ref_path)
        if isinstance(mask_path, str):
            mask_path = Path(mask_path)
        if isinstance(depth_path, str):
            depth_path = Path(depth_path)

        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path, depth_path, cos_0_path)
        else:
            video_reader = decord.VideoReader(uri=str(path))
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )
            
            step = video_num_frames // nearest_frame_bucket
            frame_indices = list(range(0, video_num_frames, step))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)
            
            # Apply color jitter to frames
            color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            frames = torch.stack([color_jitter(frame) for frame in frames], dim=0)

            ref = None
            if self.image_to_video:
                if ref_path and ref_path.is_file():
                    try:
                        if self._is_video_file(ref_path):
                            ref_video_reader = decord.VideoReader(uri=str(ref_path))
                            ref_video_num_frames = len(ref_video_reader)
                            
                            if ref_video_num_frames < nearest_frame_bucket:
                                ref_frame_indices = list(range(ref_video_num_frames))
                                ref_frames = ref_video_reader.get_batch(ref_frame_indices)
                                ref_frames = ref_frames.float()
                                ref_frames = ref_frames.permute(0, 3, 1, 2).contiguous()
                                
                                # padding 0
                                padding_frames = torch.zeros((nearest_frame_bucket - ref_video_num_frames, ref_frames.shape[1], ref_frames.shape[2], ref_frames.shape[3]))
                                ref_frames = torch.cat([ref_frames, padding_frames], dim=0)
                            else:
                                ref_step = ref_video_num_frames // nearest_frame_bucket
                                ref_frame_indices = list(range(0, ref_video_num_frames, ref_step))
                                ref_frames = ref_video_reader.get_batch(ref_frame_indices)
                                ref_frames = ref_frames[:nearest_frame_bucket].float()
                                ref_frames = ref_frames.permute(0, 3, 1, 2).contiguous()
                            
                            ref_frames_resized = torch.stack([resize(ref_frame, nearest_res) for ref_frame in ref_frames], dim=0)
                            ref = torch.stack([self.video_transforms(ref_frame) for ref_frame in ref_frames_resized], dim=0)

                        elif self._is_image_file(ref_path):
                            ref_img = Image.open(str(ref_path)).convert("RGB")
                            ref_img_tensor = transforms.ToTensor()(ref_img)
                            resized_ref_img_tensor = resize(ref_img_tensor, nearest_res)
                            ref = self.video_transforms(resized_ref_img_tensor).unsqueeze(0)

                        else:
                            logger.warning(f"Unsupported reference file format at {ref_path}: using first frame as reference")
                            ref = frames[:1].clone() # use first frame as reference

                    except Exception as e:
                        logger.error(f"Failed to load or process reference file at {ref_path}: {e}")
                        ref = frames[:1].clone() # use first frame as reference
                else:
                    ref = frames[:1].clone() # use first frame as reference
            
            tracking_reader = decord.VideoReader(uri=str(tracking_path))
            tracking_frames = tracking_reader.get_batch(frame_indices)
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

            depth_frames = None
            if depth_path and depth_path.is_file():
                depth_reader = decord.VideoReader(uri=str(depth_path))
                depth_frames = depth_reader.get_batch(frame_indices)
                depth_frames = depth_frames[:nearest_frame_bucket].float()
                depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
                depth_frames_resized = torch.stack([resize(depth_frame, nearest_res) for depth_frame in depth_frames], dim=0)
                depth_frames = torch.stack([self.video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)

            mask_frames = None
            if mask_path and mask_path.is_file():
                mask_reader = decord.VideoReader(uri=str(mask_path))
                mask_num_frames = len(mask_reader)
                
                if mask_num_frames < nearest_frame_bucket:
                    # Read all available frames
                    mask_frame_indices = list(range(mask_num_frames))
                    mask_frames = mask_reader.get_batch(mask_frame_indices)
                    mask_frames = mask_frames.float()
                    mask_frames = mask_frames.permute(0, 3, 1, 2).contiguous()
                    
                    # Resize frames first
                    mask_frames_resized = torch.stack([resize(mask_frame, nearest_res) for mask_frame in mask_frames], dim=0)
                    
                    # Pad by repeating the last frame
                    last_frame = mask_frames_resized[-1:].clone()  # Get last frame and keep batch dimension
                    padding_frames = last_frame.repeat(nearest_frame_bucket - mask_num_frames, 1, 1, 1)
                    mask_frames = torch.cat([mask_frames_resized, padding_frames], dim=0)
                else:
                    # Use the same frame_indices as main video
                    mask_frames = mask_reader.get_batch(frame_indices)
                    mask_frames = mask_frames[:nearest_frame_bucket].float()
                    mask_frames = mask_frames.permute(0, 3, 1, 2).contiguous()
                    mask_frames = torch.stack([resize(mask_frame, nearest_res) for mask_frame in mask_frames], dim=0)

            cos_frames_list = None
            if cos_0_path and cos_0_path.is_file():
                cos_frames_list = []
                for i in range(self.cos_level):
                    base_name = cos_0_path.stem.replace("_cos_i_0", "")
                    cos_path = cos_0_path.parent.joinpath(f"{base_name}_cos_i_{i}.mp4")
                    
                    if cos_path.is_file():
                        cos_reader = decord.VideoReader(uri=str(cos_path))
                        cos_frames = cos_reader.get_batch(frame_indices)
                        cos_frames = cos_frames[:nearest_frame_bucket].float()
                        cos_frames = cos_frames.permute(0, 3, 1, 2).contiguous()
                        cos_frames_resized = torch.stack([resize(cos_frame, nearest_res) for cos_frame in cos_frames], dim=0)
                        cos_frames = torch.stack([self.video_transforms(cos_frame) for cos_frame in cos_frames_resized], dim=0)
                        cos_frames_list.append(cos_frames)
                    else:
                        cos_frames_list.append(None) 

            return ref, frames, tracking_frames, depth_frames, cos_frames_list, mask_frames

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        tracking_path = self.data_root.joinpath(self.tracking_column)
        
        ref_paths_list = []
        if self.ref_column:
            ref_file_path = self.data_root.joinpath(self.ref_column)
            if not ref_file_path.exists() or not ref_file_path.is_file():
                logger.warning(
                    f"Expected `{self.ref_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to reference files, but it was not found or is not a file. Will proceed without reference files from this source."
                )
            else:
                with open(ref_file_path, "r", encoding="utf-8") as file:
                    ref_paths_list = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
                    if any(not path.is_file() for path in ref_paths_list):
                        logger.warning(
                            f"Expected all paths in `{self.ref_column}` file to be valid files, but found at least one invalid path. Will attempt to use valid paths."
                        )

        depth_paths_list = []
        if self.depth_column:
            depth_file_path = self.data_root.joinpath(self.depth_column)
            if not depth_file_path.exists() or not depth_file_path.is_file():
                logger.warning(
                    f"Expected `{self.depth_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to depth files, but it was not found or is not a file. Will proceed without depth files from this source."
                )
            else:
                with open(depth_file_path, "r", encoding="utf-8") as file:
                    depth_paths_list = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
                    if any(not path.is_file() for path in depth_paths_list):
                        logger.warning(
                            f"Expected all paths in `{self.depth_column}` file to be valid files, but found at least one invalid path. Will attempt to use valid paths."
                        )

        cos_0_paths_list = []
        if self.cos_column:
            cos_0_file_path = self.data_root.joinpath(self.cos_column)
            if not cos_0_file_path.exists() or not cos_0_file_path.is_file():
                logger.warning(
                    f"Expected `{self.cos_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to cos_0 files, but it was not found or is not a file. Will proceed without cos_0 files from this source."
                )
            else:
                with open(cos_0_file_path, "r", encoding="utf-8") as file:
                    cos_0_paths_list = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
                    if any(not path.is_file() for path in cos_0_paths_list):
                        logger.warning(
                            f"Expected all paths in `{self.cos_column}` file to be valid files, but found at least one invalid path. Will attempt to use valid paths."
                        )

        
        densitys_list = []
        if self.density_column:
            density_file_path = self.data_root.joinpath(self.density_column)
            if not density_file_path.exists() or not density_file_path.is_file():
                logger.warning(
                    f"Expected `{self.density_column=}` to be a path to a file in `{self.data_root=}` containing line-separated integer density values (1 or greater), but it was not found or is not a file. Will proceed without density values from this source."
                )
            else:
                with open(density_file_path, "r", encoding="utf-8") as file:
                    densitys_list = []
                    for line_num, line in enumerate(file.readlines(), 1):
                        line = line.strip()
                        if len(line) > 0:
                            try:
                                density_value = int(line)
                                if density_value < 1:
                                    logger.warning(
                                        f"Expected density value to be 1 or greater, but found {density_value} at line {line_num} in `{self.density_column}` file. Skipping this line."
                                    )
                                    continue
                                densitys_list.append(density_value)
                            except ValueError:
                                logger.warning(
                                    f"Expected density value to be an integer, but found '{line}' at line {line_num} in `{self.density_column}` file. Skipping this line."
                                )
                                continue


        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not tracking_path.exists() or not tracking_path.is_file():
            raise ValueError(
                "Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated tracking information."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(tracking_path, "r", encoding="utf-8") as file:
            tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        if ref_paths_list:
            if len(ref_paths_list) != len(prompts):
                 raise ValueError(
                    f"Expected length of prompts and reference files to be the same but found {len(prompts)=} and {len(ref_paths_list)=}."
                 )
            self.ref_paths = ref_paths_list
        else:
            self.ref_paths = [None] * len(prompts)

        if depth_paths_list:
            if len(depth_paths_list) != len(prompts):
                raise ValueError(
                    f"Expected length of prompts and depth files to be the same but found {len(prompts)=} and {len(depth_paths_list)=}."
                )
            self.depth_paths = depth_paths_list
        else:
            self.depth_paths = [None] * len(prompts)

        if cos_0_paths_list:
            if len(cos_0_paths_list) != len(prompts):
                raise ValueError(
                    f"Expected length of prompts and cos_0 files to be the same but found {len(prompts)=} and {len(cos_0_paths_list)=}."
                )
            self.cos_0_paths = cos_0_paths_list
        else:
            self.cos_0_paths = [None] * len(prompts)

        if densitys_list:
            if len(densitys_list) != len(prompts):
                raise ValueError(
                    f"Expected length of prompts and density values to be the same but found {len(prompts)=} and {len(densitys_list)=}."
                )
            self.densitys = densitys_list
        else:
            self.densitys = [None] * len(prompts)

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if self.tracking_column and self.tracking_column in df.columns:
            tracking_paths = df[self.tracking_column].tolist()
            tracking_paths = [self.data_root.joinpath(line.strip()) for line in tracking_paths]
            self.tracking_paths = tracking_paths
        else:
            raise ValueError(f"Expected tracking column '{self.tracking_column}' not found in CSV file.")


        if self.ref_column and self.ref_column in df.columns:
            self.ref_paths = [self.data_root.joinpath(line.strip()) if pd.notna(line) and line.strip() else None for line in df[self.ref_column].tolist()]
        else:
            self.ref_paths = [None] * len(prompts)
        
        if self.mask_column and self.mask_column in df.columns:
            self.mask_paths = [self.data_root.joinpath(line.strip()) if pd.notna(line) and line.strip() else None for line in df[self.mask_column].tolist()]
        else:
            self.mask_paths = [None] * len(prompts)
        
        if self.depth_column and self.depth_column in df.columns:
            self.depth_paths = [self.data_root.joinpath(line.strip()) if pd.notna(line) and line.strip() else None for line in df[self.depth_column].tolist()]
        else:
            self.depth_paths = [None] * len(prompts)

        if self.cos_column and self.cos_column in df.columns:
            self.cos_0_paths = [self.data_root.joinpath(line.strip()) if pd.notna(line) and line.strip() else None for line in df[self.cos_column].tolist()]
        else:
            self.cos_0_paths = [None] * len(prompts)

        if self.density_column and self.density_column in df.columns:
            self.densitys = []
            for line in df[self.density_column].tolist():
                if pd.notna(line):
                    try:
                        density_value = int(line)
                        if density_value < 1:
                            logger.warning(
                                f"Expected density value to be 1 or greater, but found {density_value} in CSV column '{self.density_column}'. Skipping this value."
                            )
                            self.densitys.append(None)
                        else:
                            self.densitys.append(density_value)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Expected density value to be an integer, but found '{line}' in CSV column '{self.density_column}'. Skipping this value."
                        )
                        self.densitys.append(None)
                else:
                    self.densitys.append(None)
        else:
            self.densitys = [None] * len(prompts)


        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected video paths in column '{self.video_column}' of '{self.dataset_file}' to be valid files relative to '{self.data_root}'. Check paths like '{video_paths[0] if video_paths else 'N/A'}'."
            )

        if any(not path.is_file() for path in self.tracking_paths):
            raise ValueError(
                f"Expected tracking paths in column '{self.tracking_column}' of '{self.dataset_file}' to be valid files relative to '{self.data_root}'. Check paths like '{self.tracking_paths[0] if self.tracking_paths else 'N/A'}'."
            )

        return prompts, video_paths
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        ref_path_for_item = None
        if hasattr(self, 'ref_paths') and self.ref_paths and index < len(self.ref_paths):
            ref_path_for_item = self.ref_paths[index]
        
        mask_path_for_item = None
        if hasattr(self, 'mask_paths') and self.mask_paths and index < len(self.mask_paths):
            mask_path_for_item = self.mask_paths[index]

        depth_path_for_item = None
        if hasattr(self, 'depth_paths') and self.depth_paths and index < len(self.depth_paths):
            depth_path_for_item = self.depth_paths[index]

        cos_0_path_for_item = None
        if hasattr(self, 'cos_0_paths') and self.cos_0_paths and index < len(self.cos_0_paths):
            cos_0_path_for_item = self.cos_0_paths[index]

        density_value_for_item = None
        if hasattr(self, 'densitys') and self.densitys and index < len(self.densitys):
            if self.densitys[index] is not None:
                density_value_for_item = 1 / self.densitys[index]

        if self.load_tensors:
            ref_latents, video_latents, tracking_map, depth_latents, cos_latents_list, _ = self._preprocess_video(self.video_paths[index], self.tracking_paths[index], ref_path=ref_path_for_item, mask_path=mask_path_for_item, depth_path=depth_path_for_item, cos_0_path=cos_0_path_for_item)

            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "text": self.id_token + self.prompts[index],
                "ref": ref_latents,
                "video": video_latents,
                "tracking_map": tracking_map,
                "depth": depth_latents,
                "cos_latents_list": cos_latents_list,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            ref, video, tracking_map, depth, cos_frames_list, mask_video_input = self._preprocess_video(self.video_paths[index], self.tracking_paths[index], ref_path=ref_path_for_item, mask_path=mask_path_for_item, depth_path=depth_path_for_item, cos_0_path=cos_0_path_for_item)
            sample = {
                "text": self.id_token + self.prompts[index],
                "clip_pixel_values": ref,
                "pixel_values": video,
                "control_pixel_values": tracking_map,
                "depth_pixel_values": depth,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
                "density": density_value_for_item,
            }

            if cos_frames_list is not None:
                for i in range(self.cos_level):
                    sample[f"cos_pixel_values_{i}"] = cos_frames_list[i]

            if self.enable_inpaint:
                if mask_video_input is not None:
                    mask = generate_mask(mask_video_input)
                else:
                    mask = get_random_mask(video.size())
                mask_pixel_values = video * (1 - mask) + torch.ones_like(video) * -1 * mask
                sample["mask_pixel_values"] = mask_pixel_values
                sample["mask"] = mask

                if ref is not None:
                    clip_pixel_values = ref[0].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    sample["clip_pixel_values"] = clip_pixel_values
                    sample["clip_idx"] = 0

                    ref_pixel_values = ref
                    if (mask == 1).all():
                        ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
                    sample["ref_pixel_values"] = ref_pixel_values
            
            return sample
    
    def _load_preprocessed_latents_and_embeds(self, path: Path, tracking_path: Path, depth_path: Optional[Path] = None, cos_0_path: Optional[Path] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        tracking_map_path = path.parent.parent.joinpath("tracking_map")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or not tracking_map_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents`, `prompt_embeds`, and `tracking_map`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        tracking_map_filepath = tracking_map_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file() or not tracking_map_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = str(image_latent_filepath)
            video_latent_filepath = str(video_latent_filepath)
            tracking_map_filepath = str(tracking_map_filepath)
            embeds_filepath = str(embeds_filepath)
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} or {tracking_map_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        ref_latents = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        tracking_map = torch.load(tracking_map_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        if depth_path is not None:
            depth_latents_path = path.parent.parent.joinpath("depth_latents")
            depth_latent_filepath = depth_latents_path.joinpath(pt_filename)
            if not depth_latent_filepath.is_file():
                raise ValueError(f"The file {depth_latent_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`.")
            depth_latents = torch.load(depth_latent_filepath, map_location="cpu", weights_only=True)
        else:
            depth_latents = None

        cos_latents_list = None

        return ref_latents, latents, tracking_map, depth_latents, cos_latents_list, embeds

class CollateFunctionTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool, enable_inpaint: bool, cos_level: int) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors
        self.enable_inpaint = enable_inpaint
        self.cos_level = cos_level

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["text"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["clip_pixel_values"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        clip_idx = [x["clip_idx"] for x in data[0]]

        videos = [x["pixel_values"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        control_pixel_values = [x["control_pixel_values"] for x in data[0]]
        control_pixel_values = torch.stack(control_pixel_values).to(dtype=self.weight_dtype, non_blocking=True)

        # Handle depth data if available
        depth_pixel_values = None
        if "depth_pixel_values" in data[0][0]:
            depth_pixel_values = [x["depth_pixel_values"] for x in data[0]]
            depth_pixel_values = torch.stack(depth_pixel_values).to(dtype=self.weight_dtype, non_blocking=True)

        # Handle cos data if available
        cos_pixel_values_list = []
        for i in range(self.cos_level):
            cos_key = f"cos_pixel_values_{i}"
            if cos_key in data[0][0] and data[0][0][cos_key] is not None:
                cos_pixel_values = [x[cos_key] for x in data[0]]
                cos_pixel_values = torch.stack(cos_pixel_values).to(dtype=self.weight_dtype, non_blocking=True)
                cos_pixel_values_list.append(cos_pixel_values)
            else:
                cos_pixel_values_list.append(None)

        density = [x["density"] for x in data[0]]
        if self.enable_inpaint:
            mask = [x["mask"] for x in data[0]]
            mask = torch.stack(mask).to(dtype=self.weight_dtype, non_blocking=True)
            ref_pixel_values = [x["ref_pixel_values"] for x in data[0]]
            ref_pixel_values = torch.stack(ref_pixel_values).to(dtype=self.weight_dtype, non_blocking=True)
            mask_pixel_values = [x["mask_pixel_values"] for x in data[0]]
            mask_pixel_values = torch.stack(mask_pixel_values).to(dtype=self.weight_dtype, non_blocking=True)

            result = {
                "clip_pixel_values": images,
                "clip_idx":  torch.tensor(clip_idx),
                "pixel_values": videos,
                "text": prompts,
                "control_pixel_values": control_pixel_values,
                "mask": mask,
                "ref_pixel_values": ref_pixel_values,
                "mask_pixel_values": mask_pixel_values,
                "density": torch.tensor(density),
            }
            
            if depth_pixel_values is not None:
                result["depth_pixel_values"] = depth_pixel_values
            
            # Add cos data to result
            for i, cos_pixel_values in enumerate(cos_pixel_values_list):
                if cos_pixel_values is not None:
                    result[f"cos_pixel_values_{i}"] = cos_pixel_values
                
            return result
            
        result = {
            "clip_pixel_values": images,
            "clip_idx":  torch.tensor(clip_idx),
            "pixel_values": videos,
            "text": prompts,
            "control_pixel_values": control_pixel_values,
            "density": torch.tensor(density),
        }
        
        if depth_pixel_values is not None:
            result["depth_pixel_values"] = depth_pixel_values
        
        # Add cos data to result
        for i, cos_pixel_values in enumerate(cos_pixel_values_list):
            if cos_pixel_values is not None:
                result[f"cos_pixel_values_{i}"] = cos_pixel_values
            
        return result
