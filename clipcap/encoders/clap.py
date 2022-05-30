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
    def __init__(self, sample_rate: int = 48000, max_duration: float = 10.0, num_windows: Optional[int] = None,
                 window_overlap_percentage: float = 0.0) -> None:
        import torchaudio.functional as F
        import soundfile as sf

        self.resampler = F.resample
        self.loader = sf.read

        self.sample_rate = sample_rate
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
        if waveform.shape[0] > max_waveform_size:
            waveform = waveform[:max_waveform_size]

        # Size = the length of each patch in samples
        size = math.ceil(waveform.shape[0] / self.num_windows)

        # Pad to nearest sample to avoid the end being cut off.
        pad_amount = (size * self.num_windows) - waveform.shape[0]
        if pad_amount != 0:
            waveform = nnf.pad(
                waveform,
                (0, pad_amount),
                mode="constant",
                value=0,
            )

        # Step = the gap between each patch in samples
        if self.window_overlap_percentage != 0.0:
            step = math.floor(size * (1 - (self.window_overlap_percentage / 100)))
        else:
            step = size
        
        # Unfold waveform into patches of size `size` with stride `step` between each patch.
        tiles = waveform.unfold(0, size, step)

        # If we weren't able to fit the max sample size in each tile, we must now pad to make the sizes consistant.
        if tiles.shape[1] < self.max_samples:
            tiles = nnf.pad(
                tiles,
                (0, self.max_samples - tiles.shape[1], 0, 0),
                mode="constant",
                value=0,
            )

        # If an overlap is enabled, we must remove the extra data from the unfold so that it only as `num_windows` samples.
        if tiles.shape[0] > self.num_windows:
            tiles = tiles[:self.num_windows, :]
        
        return tiles
   
    def __call__(self, file: Union[BytesIO, str, bytes, os.PathLike]) -> torch.Tensor:
        waveform, file_sample_rate = self.loader(file, dtype='float32')
        waveform = torch.from_numpy(waveform)

        # Convert to mono.
        waveform = torch.mean(waveform, dim=0)
        
        # Resample if neccesary to produce a consistant sample rate.
        if file_sample_rate != self.sample_rate:
            waveform = self.resampler(waveform, file_sample_rate, self.sample_rate)
        
        # Produce global audio sample.
        if waveform.shape[0] > self.max_samples:
            new_waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            new_waveform = nnf.pad(
                waveform,
                (0, self.max_samples - waveform.shape[0]),
                mode="constant",
                value=0,
            )

        # Produce tiled audio samples from the original waveform, if enabled.
        if self.num_windows is not None:
            new_waveform = torch.cat((
                new_waveform.unsqueeze(0), self.tile_waveform(waveform)
            ), dim=0)
        
        return new_waveform


def get_clap_encoder(model_path: str, window_size: Optional[int] = None,
                     window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Module, Callable]:
    import open_clip # where open_clip = the LAION-AI/CLAP fork
    from collections import OrderedDict
    
    transform = CLAPTransform(
        sample_rate=48000,
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

    def _encode_fn(x: torch.Tensor) -> torch.Tensor:
        # 'Hack' to retain tiled audio inputs in the same batch in CLAP.
        original_shape = x.shape
        
        if window_size is not None:
            # Flatten to allow patches to be inputted into CLAP.
            x = torch.flatten(x, start_dim=0, end_dim=1)
        
        out = model.encode_audio(x)["embedding"]
        
        if window_size is not None:
            # Unflatten
            out = out.view(original_shape[0], original_shape[1], *out[1:])

        return out


    return _encode_fn, transform