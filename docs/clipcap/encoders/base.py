from clipcap.encoders.clip import get_clip_encoder
from clipcap.encoders.clap import get_clap_encoder

from typing import Tuple, Callable, Optional
from argparse import ArgumentParser
from torch.nn import Module

def get_encoder(encoder_model_name: str, encoder_model_variant: str, window_size: Optional[int] = None, 
                window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Module, Callable]:
    kwargs = {
        "window_size": window_size,
        "window_overlap_percentage": window_overlap_percentage,
        "device": device
    }

    if encoder_model_name == "clip":
        return get_clip_encoder(encoder_model_variant, **kwargs)
    elif encoder_model_name == "clap":
        return get_clap_encoder(encoder_model_variant, **kwargs)
    else:
        # Feel free to raise an issue / PR if your desired model is not supported
        raise ValueError(f"invalid encoder name: '{encoder_model_name}'")


def get_encoder_from_args(args: ArgumentParser) -> Tuple[Module, Callable]:
    return get_encoder(
        args.encoder_model_name, args.encoder_model_variant, window_size=args.window_size,
        window_overlap_percentage=args.window_overlap_percentage, device=args.device
    )