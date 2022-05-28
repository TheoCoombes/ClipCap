from torch.nn import Module as Module
from typing import Tuple, Callable
from io import BytesIO
import torch

class CLIPTransform(object):
    def __init__(self, clip_preprocess: Callable):
        from PIL import Image

        self.loader = Image.open
        self.clip_preprocess = clip_preprocess
    
    def __call__(self, file: BytesIO) -> torch.Tensor:
        image = self.loader(file)
        image_tensor = self.clip_preprocess(image)
        return image_tensor


def get_clip_encoder(encoder_model_variant: str, device: str = "cuda") -> Tuple[Module, Callable]:
    import clip

    model, preprocess = clip.load(encoder_model_variant, device=device)

    transform = CLIPTransform(preprocess)

    return model.encode_image, transform