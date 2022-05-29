# Reference format for modular encoders (useful for creating PRs).

from typing import Tuple, Callable, Optional
from torch.nn import Module
from io import BytesIO
import torch


class BaseEncoderTransform(object):
    def __init__(self, foo: int, bar: str = "wizz"):
        # Imports go here, referenced as class attributes, e.g.
        # import medialibrary
        # self.loader = medialibrary.load
        
        def _donothing(*args):
            return None
        self.loader = _donothing

        self.foo = foo
        self.bar = bar
    
    def __call__(self, file: BytesIO) -> torch.Tensor:
        # Always inputted with a BytesIO object for modularity.

        media = self.loader(file, self.foo, self.bar)
        media_tensor = torch.tensor(media)
    
        return media_tensor

def get_base_encoder(model_specific_variable: str, window_size: Optional[int] = None, device: str = "cuda") -> Tuple[Module, Callable]:
    # This function then gets plugged into `get_encoder(...)` at `base.py` where this function takes a single string argument and device string as params.
    # For model-specific config, perhaps it's best to implement a dictionary somewhere? e.g. {model_variant: {config_dict...}}
    # `model_specific_variable` can be anything tailored best to the model, e.g. a path to the checkpoint or HuggingFace transformers model name.
    # `window_size` is optional and does no neccesarily need to be supported for the model, however, allows for tiling of the content allowing more \
    #    fine-grained detail to be inputted into the LM. See `clip.py` for an example of this implementation.

    # from encoderlibrary import Encoder
    Encoder = object # placeholder for demo (see above for example)
    
    transform = BaseEncoderTransform(123, bar="args...")
    encoder = Encoder(pretrained_path=model_specific_variable).to(device).encode_media # must refer to the specific media of the encoder (i.e. .encode_image for CLIP / .encode_audio for CLAP)

    return encoder, transform