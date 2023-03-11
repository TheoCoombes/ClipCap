from clipcap.encoders.clip import get_clip_encoder
from clipcap.encoders.clap import get_clap_encoder
from clipcap.encoders.config import EncoderConfig

from clipcap.model import ClipCapModel, ClipCapModelPrefixOnly

from typing import Tuple, Callable, Optional, Union
from torch.nn import Module

def get_encoder(encoder_model_name: str, encoder_model_variant: str, normalize_embeddings: bool = False, window_size: Optional[int] = None, 
                use_windowed_embeddings: bool = False, window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Module, Callable]:
    kwargs = {
        "normalize_embeddings": normalize_embeddings,
        "device": device
    }

    if encoder_model_name == "clip":
        return get_clip_encoder(encoder_model_variant, use_windowed_embeddings=use_windowed_embeddings,
                                window_size=window_size, window_overlap_percentage=window_overlap_percentage,
                                **kwargs)
    elif encoder_model_name == "clap":
        return get_clap_encoder(**kwargs)
    else:
        # Feel free to raise an issue / PR if your desired model is not supported
        raise ValueError(f"invalid encoder name: '{encoder_model_name}'")


def get_encoder_from_config(config: EncoderConfig, device: str = "cpu") -> Tuple[Module, Callable]:
    if config.encoder_model_name == "clip":
        config.encoder_model_variant = config.encoder_model_variant.replace("_", "/")
    
    return get_encoder(
        config.encoder_model_name, config.encoder_model_variant, normalize_embeddings=config.normalize_embeddings,
        use_windowed_embeddings=config.use_windowed_embeddings, window_size=config.window_size,
        window_overlap_percentage=config.window_overlap_percentage, device=device
    )

def get_encoder_from_model(model: Union[ClipCapModel, ClipCapModelPrefixOnly], device: str = "cpu") -> Tuple[Module, Callable]:
    return get_encoder_from_config(model.config.encoder_config, device=device)
