from ClipCap.encoders.clip import get_clip_encoder
from ClipCap.encoders.clap import get_clap_encoder
from ClipCap.encoders.args import add_encoder_args

from typing import (
    Tuple as _Tuple,
    Callable as _Callable
)
from argparse import ArgumentParser as _ArgumentParser
from torch.nn import Module as _Module

def get_encoder(encoder_model_name: str, encoder_model_variant: str, device: str = "cuda") -> _Tuple[_Module, _Callable]:
    if encoder_model_name == "clip":
        return get_clip_encoder(encoder_model_variant, device=device)
    elif encoder_model_name == "clap":
        return get_clap_encoder(encoder_model_variant, device=device)
    else:
        # Feel free to raise an issue / PR if your desired model is not supported :)
        raise ValueError(f"invalid encoder name: '{encoder_model_name}'")

def get_encoder_from_args(args: _ArgumentParser) -> _Tuple[_Module, _Callable]:
    return get_encoder(args.encoder_model_name, args.encoder_model_variant, args.device)