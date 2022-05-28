from typing import Tuple, Callable
import torch

def get_clip_encoder(encoder_model_variant: str, device: str = "cuda") -> Tuple[torch.Module, Callable]:
    from PIL import Image
    import clip

    model, preprocess = clip.load(encoder_model_variant, device=device)

    transform = lambda file: preprocess(Image.open(file))

    return model.encode_image, transform