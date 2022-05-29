from typing import Tuple, Callable, Optional, Union
from torch.nn import functional as nnf
from torch.nn import Module
from io import BytesIO
import torch
import math
import os

_CONFIG = {
    "embed_dim": 768,
    "audio_cfg": {
        "audio_length": 1024,
        "clip_samples": 480000,
        "mel_bins": 64,
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "fmin": 50,
        "fmax": 14000,
        "class_num": 527,
        "model_type": "HTSAT",
        "model_name": "tiny"
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 512,
        "heads": 8,
        "layers": 12
    }
}

class CLAPTransform(object):
    def __init__(self, sample_rate: int = 48000, mono: bool = True, max_duration: float = 10.0,
                 num_windows: Optional[int] = None, window_overlap_percentage: float = 0.0) -> None:
        import torchaudio.functional as F
        import torchaudio

        self.resampler = F.resample
        self.loader = torchaudio.load

        self.sample_rate = sample_rate
        self.mono = mono
        self.max_samples = math.floor(max_duration * sample_rate)

        self.num_windows = num_windows
        self.window_overlap_percentage = window_overlap_percentage
    
    def tile_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Splice waveform `waveform` into `self.num_windows` patches overlapping by `self.window_overlap_percentage` percent.
        See: https://i.stack.imgur.com/uCrOg.png
        """

        # We want all resulting patches to be <= `self.max_samples` long (before padding).
        max_waveform_size = self.max_samples * self.num_windows
        if waveform.shape[-1] > max_waveform_size:
            waveform = waveform[:, :max_waveform_size]

        # Step = the gap between each patch in samples
        step = math.ceil(waveform.shape[-1] / self.num_windows)

        # Size = the length of each patch in samples
        if self.window_overlap_percentage != 0.0:
            size = math.ceil(step * (1 + (self.window_overlap_percentage / 100)))
        else:
            size = step

        # We must now pad the waveform so that the `waveform.unfold` method can produce `self.num_windows` patches.
        # Formula from https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch.Tensor.unfold
        required_waveform_size = math.ceil(waveform.shape[-1] / size) * size
        if waveform.shape[-1] < required_waveform_size:
            waveform = nnf.pad(
                waveform,
                (0, required_waveform_size - waveform.shape[-1], 0, 0),
                mode="constant",
                value=0,
            )
        
        # Unfold waveform into patches of size `size` with stride `step` between each patch.
        tiles = waveform.unfold(-1, size, step)

        if tiles.shape[-1] < self.max_samples:
            tiles = nnf.pad(
                tiles,
                (0, self.max_samples - tiles.shape[-1], 0, 0, 0, 0),
                mode="constant",
                value=0,
            )
        
        return tiles
   
    def __call__(self, file: Union[BytesIO, str, bytes, os.PathLike]) -> torch.Tensor:
        waveform, file_sample_rate = self.loader(file, channels_first=True)

        if self.mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if file_sample_rate != self.sample_rate:
            waveform = self.resampler(waveform, file_sample_rate, self.sample_rate)
        
        if self.num_windows is not None:
            waveform = self.tiled_preprocess(waveform)
            waveform = self.ensure_tileable(waveform)

        else:
            if waveform.shape[-1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif waveform.shape[-1] < self.max_samples:
                waveform = nnf.pad(
                    waveform,
                    (0, self.max_samples - waveform.shape[-1], 0, 0),
                    mode="constant",
                    value=0,
                )
        
        # shape = [channels, samples] = [1, samples] for mono
        
        # TODO
        # if self.window_size is not None:
        #     split_size = self.sample_rate // self.window_size
        #     waveform = torch.cat((waveform.unsqueeze(0), torch.split(waveform, split_size, dim=0)), dim=0)
        
        return waveform


def get_clap_encoder(model_path: str, window_size: Optional[int] = None,
                     window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Module, Callable]:
    import open_clip # where open_clip = the LAION-AI/CLAP fork
    from collections import OrderedDict
    
    transform = CLAPTransform(
        sample_rate=48000,
        mono=True,
        max_duration=10,
        num_windows=window_size,
        window_overlap_percentage=window_overlap_percentage
    )

    model = open_clip.openai.load_openai_model("ViT-B-16", _CONFIG, device="cpu", jit=False)
    model = model.float()

    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = OrderedDict()

    for key, value in ckpt["state_dict"].items():
        name = key[7:] # remove `module.`
        state_dict[name] = value

    model.load_state_dict(state_dict)
    model = model.to(device)

    _encode_fn = lambda x: model.encode_audio(x)["embedding"]

    return _encode_fn, transform