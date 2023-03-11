from typing import Tuple, Callable, Optional, Union
from torch.nn import functional as nnf
from io import BytesIO
from torch import nn
import numpy as np
import torch
import math
import os

class CLAPTransform(object):
    def __init__(self, sample_rate: int = 48000) -> None: # , max_duration: float = 10.0, num_windows: Optional[int] = None, use_windowed_embeddings: bool = False, window_overlap_percentage: float = 0.0
        import torchaudio.functional as F
        import soundfile as sf

        self.resampler = F.resample
        self.loader = sf.read

        self.sample_rate = sample_rate
        #self.max_samples = math.ceil(max_duration * sample_rate)

        #self.use_windowed_embeddings = use_windowed_embeddings
        #self.num_windows = num_windows
        #self.window_overlap_percentage = window_overlap_percentage
    
#     def tile_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Splice waveform `waveform` into `self.num_windows` patches overlapping by `self.window_overlap_percentage` percent.
#         See: https://i.stack.imgur.com/uCrOg.png
#         """

#         # We want all resulting patches to be <= `self.max_samples` long (before padding).
#         max_waveform_size = self.max_samples * self.num_windows
#         if waveform.shape[0] > max_waveform_size:
#             waveform = waveform[:max_waveform_size]

#         # Size = the length of each patch in samples
#         size = math.ceil(waveform.shape[0] / self.num_windows)

#         # Pad to nearest sample to avoid the end being cut off.
#         pad_amount = (size * self.num_windows) - waveform.shape[0]
#         if pad_amount != 0:
#             waveform = nnf.pad(
#                 waveform,
#                 (0, pad_amount),
#                 mode="constant",
#                 value=0,
#             )

#         # Step = the gap between each patch in samples
#         if self.window_overlap_percentage != 0.0:
#             step = math.floor(size * (1 - (self.window_overlap_percentage / 100)))
#         else:
#             step = size
        
#         # Unfold waveform into patches of size `size` with stride `step` between each patch.
#         tiles = waveform.unfold(0, size, step)

#         # If we weren't able to fit the max sample size in each tile, we must now pad to make the sizes consistant.
#         if tiles.shape[1] < self.max_samples:
#             tiles = nnf.pad(
#                 tiles,
#                 (0, self.max_samples - tiles.shape[1], 0, 0),
#                 mode="constant",
#                 value=0,
#             )

#         # If an overlap is enabled, we must remove the extra data from the unfold so that it only as `num_windows` samples.
#         if tiles.shape[0] > self.num_windows:
#             tiles = tiles[:self.num_windows, :]
        
#         return tiles
   
    def __call__(self, file: Union[BytesIO, str, bytes, os.PathLike]) -> np.ndarray:
        waveform, file_sample_rate = self.loader(file, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(waveform)

        # Convert to mono.
        waveform = torch.mean(waveform, dim=0)
        
        # Resample if neccesary to produce a consistant sample rate.
        if file_sample_rate != self.sample_rate:
            waveform = self.resampler(waveform, file_sample_rate, self.sample_rate)
        
        # Produce global audio sample.
#         if waveform.shape[0] > self.max_samples:
#             new_waveform = waveform[:self.max_samples]
#         elif waveform.shape[0] < self.max_samples:
#             new_waveform = nnf.pad(
#                 waveform,
#                 (0, self.max_samples - waveform.shape[0]),
#                 mode="constant",
#                 value=0,
#             )

#         # Produce tiled audio samples from the original waveform, if enabled.
#         if self.use_windowed_embeddings:
#             new_waveform = torch.cat((
#                 new_waveform.unsqueeze(0), self.tile_waveform(waveform)
#             ), dim=0)

        waveform = waveform.numpy() # must be numpy array
        
        return waveform

class CLAPModel(nn.Module):
    def __init__(self, model: nn.Module, normalize_embeddings: bool = False) -> None:
        super().__init__()
        self.model = model
        self.normalize_embeddings = normalize_embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 'Hack' to retain tiled patch inputs in the same batch in CLIP.
#         original_shape = x.shape
        
#         if self.use_windowed_embeddings:
#             # Flatten
#             x = torch.flatten(x, start_dim=0, end_dim=1)
        device = x.device
        x = x.numpy()
        
        out = self.model.get_audio_embedding_from_data(x=x)
        out = torch.from_numpy(out).to(device)
        
        if self.normalize_embeddings:
            out /= out.norm(dim=-1, keepdim=True)
        
#         if self.use_windowed_embeddings:
#             # Unflatten
#             out = out.view(original_shape[0], original_shape[1], *out.shape[1:])

        return out

def get_clap_encoder(normalize_embeddings: bool = False, device: str = "cuda") -> Tuple[Callable, Callable]:
    import laion_clap
    
    model_id = int(model_id)
    
    transform = CLAPTransform(
        sample_rate=48000
#         max_duration=10,
#         use_windowed_embeddings=use_windowed_embeddings,
#         num_windows=window_size,
#         window_overlap_percentage=window_overlap_percentage
    )

    clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=device) # Test without fusion?
    clap_model.load_ckpt()

    model = CLAPModel(
        clap_model,
        normalize_embeddings=normalize_embeddings,
        use_windowed_embeddings=use_windowed_embeddings
    )

    model = model.eval()
    model = model.to(device)

    return model, transform
