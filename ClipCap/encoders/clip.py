from torch.nn import Module as Module
from typing import Tuple, Callable, Optional, List
from io import BytesIO
from PIL import Image
import torch
import math

class CLIPTransform(object):
    def __init__(self, clip_preprocess: Callable, window_size: Optional[int] = None, window_overlap_percentage: float = 0.0):
        assert math.sqrt(window_size).is_integer(), "`window_size` must be a square number with CLIP, e.g. (3x3) = 9 for tiles of 3 by 3."

        self.loader = Image.open
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
    
    def tile_image(self, image: Image.Image) -> List[Image.Image]:
        size, _ = image.size # Image is already 1:1 by now.
        pixels_per_tile = size // self.window_size

        if self.window_overlap_percentage != 0:
            overlap_amount = math.ceil((self.window_overlap_percentage / pixels_per_tile) / 2)
        else:
            overlap_amount = 0

        crops = []

        for x in range(0, size, pixels_per_tile):
            for y in range(0, size, pixels_per_tile):
                start_x = x
                x_length = pixels_per_tile
                start_y = y
                y_length = pixels_per_tile

                if self.window_overlap_percentage != 0:
                    # Width overlap
                    if start_x >= overlap_amount:
                        start_x -= overlap_amount
                    if (size - x_length) >= overlap_amount:
                        x_length += overlap_amount

                    # Height overlap
                    if start_y >= overlap_amount:
                        start_y -= overlap_amount
                    if (size - y_length) >= overlap_amount:
                        y_length += overlap_amount
                
                crop = (start_x, start_y, x_length, y_length)
                cropped_image = image.crop(crop)
                crops.append(cropped_image)

        return crops
    
    def __call__(self, file: BytesIO) -> torch.Tensor:
        image = self.loader(file)

        if self.window_size is not None:
            image = self.center_crop(image) # Image is now squared.
            image = self.ensure_tileable(image) # Image is now divisible into window_size tiles.
            patches = self.tile_image(image)
            inputs = [image, *patches]
        else:
            inputs = [image]

        tensors = [self.clip_preprocess(input).unsqueeze(0) for input in inputs]
        
        if len(tensors) == 1:
            image_tensor = torch.flatten(tensors[0], end_dim=1)
        else:
            image_tensor = torch.cat(tensors, dim=0)

        return image_tensor


def get_clip_encoder(encoder_model_variant: str, window_size: Optional[int] = None, 
                     window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Module, Callable]:
    import clip

    model, preprocess = clip.load(encoder_model_variant, device=device)

    transform = CLIPTransform(preprocess, window_size=window_size, window_overlap_percentage=window_overlap_percentage)

    def _encode_fn(x: torch.Tensor) -> torch.Tensor:
        # 'Hack' to retain tiled patch inputs in the same batch in CLIP.
        original_shape = x.shape
        
        if window_size is not None:
            # Flatten
            single_dim = original_shape[0] * original_shape[1]
            x = x.view(single_dim, *original_shape[2:])
        
        out = model.encode_image(x)
        
        if window_size is not None:
            # Unflatten
            out = out.view(original_shape[0], original_shape[1], *out[1:])

        return out

    return _encode_fn, transform