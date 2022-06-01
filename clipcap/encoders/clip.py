from typing import Tuple, Callable, Optional, List, Union
from torch.nn import Module
from io import BytesIO
from PIL import Image
import torch
import math
import os

class CLIPTransform(object):
    def __init__(self, clip_preprocess: Callable, use_windowed_embeddings: bool = False,
                 window_size: Optional[int] = (3 * 3), window_overlap_percentage: float = 0.0) -> None:
        from torchvision import transforms

        if use_windowed_embeddings:
            assert math.sqrt(window_size).is_integer(), "`window_size` must be a square number with CLIP, e.g. (3x3) = 9 for tiles of 3 by 3."

        self.loader = Image.open
        if window_overlap_percentage != 0.0:
            self.img_to_tensor = transforms.ToTensor()
            n_px = clip_preprocess.transforms[0].size
            self.clip_transform = transforms.Compose([
                transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

        else:
            self.img_to_tensor = None
            self.clip_transform = None

        self.use_windowed_embeddings = use_windowed_embeddings
        self.window_size = window_size
        self.window_overlap_percentage = window_overlap_percentage
        self.clip_preprocess = clip_preprocess
    
    def center_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        
        if width > height:
            crop_size = (width - height) // 2
            crop = (crop_size, 0, height, height)
            image = image.crop(crop)
        elif height > width:
            crop_size = (height - width) // 2
            crop = (0, crop_size, width, width)
            image = image.crop(crop)
        
        return image
    
    def ensure_tileable(self, image: Image.Image) -> Image.Image:
        size, _ = image.size # Image is already 1:1 by now.
        tiles_per_axis = int(math.sqrt(self.window_size))
        
        target_size = math.ceil(size / tiles_per_axis) * tiles_per_axis

        if target_size != size:
            image = image.resize((target_size, target_size), resample=Image.BILINEAR)
        
        return image
    
    def tile_image(self, image: Image.Image) -> torch.Tensor:
        size, _ = image.size # Image is already 1:1 by now.
        tiles_per_axis = int(math.sqrt(self.window_size))
        pixels_per_tile = size // tiles_per_axis

        if self.window_overlap_percentage != 0:
            step = math.floor(pixels_per_tile * (1 - (self.window_overlap_percentage / 100)))
        else:
            step = pixels_per_tile

        image = image.convert("rgb")
        tensor = self.img_to_tensor(image)

        tiles = tensor.unfold(1, pixels_per_tile, step).unfold(2, pixels_per_tile, step)

        if tiles.shape[0] > tiles_per_axis:
            tiles = tiles[:tiles_per_axis, :, :, :, :]
        if tiles.shape[1] > tiles_per_axis:
            tiles = tiles[:, :tiles_per_axis, :, :, :]

        return tiles

    
    def __call__(self, file: Union[BytesIO, str, bytes, os.PathLike]) -> torch.Tensor:
        image = self.loader(file)

        if self.use_windowed_embeddings:
            image = self.center_crop(image) # Image is now squared.
            image = self.ensure_tileable(image) # Image is now divisible into window_size tiles.
            patches = self.tile_image(image) # Create patches from tileable image.
            patches = self.clip_transform(patches) # Run the remainder of the CLIP transform over it, i.e. resizing and normalizing
        else:
            patches = None

        global_tensor = self.clip_preprocess(image)
        
        if patches is not None:
            image_tensor = torch.cat((
                global_tensor.unsqueeze(0), torch.flatten(patches, start_dim=0, end_dim=1)
            ), dim=0)
        else:
            image_tensor = global_tensor

        return image_tensor

class ClipModel(Module):
    def __init__(self, model: Module, normalize_embeddings: bool = False, use_windowed_embeddings: bool = False) -> None:
        super().__init__()
        self.model = model
        self.normalize_embeddings =normalize_embeddings
        self.use_windowed_embeddings = use_windowed_embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 'Hack' to retain tiled patch inputs in the same batch in CLIP.
        original_shape = x.shape
        
        if self.use_windowed_embeddings:
            # Flatten
            x = torch.flatten(x, start_dim=0, end_dim=1)
        
        out = self.model.encode_image(x)

        if self.normalize_embeddings:
            out /= out.norm(dim=-1, keepdim=True)
        
        if self.use_windowed_embeddings:
            # Unflatten
            out = out.view(original_shape[0], original_shape[1], *out.shape[1:])

        return out


def get_clip_encoder(encoder_model_variant: str, window_size: Optional[int] = None, normalize_embeddings: bool = False,
                     use_windowed_embeddings: bool = False, window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Callable, Callable]:
    import clip

    clip_model, preprocess = clip.load(encoder_model_variant, device=device)

    transform = CLIPTransform(
        preprocess,
        use_windowed_embeddings=use_windowed_embeddings,
        window_size=window_size,
        window_overlap_percentage=window_overlap_percentage
    )

    model = ClipModel(
        clip_model,
        normalize_embeddings=normalize_embeddings,
        use_windowed_embeddings=use_windowed_embeddings
    )

    model = model.eval()
    model = model.to(device)

    return model, transform