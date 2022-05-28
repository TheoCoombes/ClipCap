from torch.nn import functional as nnf
from torch.nn import Module
from typing import Tuple, Callable, Optional
from io import BytesIO
import torch

class CLAPTransform(object):
    def __init__(self, sample_rate: int = 48000, mono: bool = True, max_duration: float = 10.0):
        import torchaudio.functional as F
        import torchaudio

        self.resampler = F.resample
        self.loader = torchaudio.load

        # if window_size is not None:
        #     assert sample_rate % window_size == 0, f"`sample_rate` ({sample_rate}) must be divisible with no remainder by `window_size` ({window_size})"

        self.sample_rate = sample_rate
        self.mono = mono
        self.max_samples = int(max_duration * sample_rate)
        # self.window_size = window_size
    
    def __call__(self, file: BytesIO) -> torch.Tensor:
        waveform, file_sample_rate = self.loader(file, channels_first=True)

        if self.mono:
            waveform = torch.mean(waveform, dim=0)
        
        if file_sample_rate != self.sample_rate:
            waveform = self.resampler(waveform, file_sample_rate, self.sample_rate)
        
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            waveform = nnf.pad(
                waveform,
                (0, self.max_samples - len(waveform)),
                mode="constant",
                value=0,
            )
        
        # TODO
        # if self.window_size is not None:
        #     split_size = self.sample_rate // self.window_size
        #     waveform = torch.cat((waveform.unsqueeze(0), torch.split(waveform, split_size, dim=0)), dim=0)
        
        return waveform


def get_clap_encoder(model_path: str, device: str = "cuda") -> Tuple[Module, Callable]:
    from open_clip import CLAP # where open_clip = the LAION-AI/CLAP fork
    from open_clip.model import CLAPAudioCfp, CLAPTextCfg
    
    transform = CLAPTransform(
        sample_rate=48000,
        mono=True,
        max_duration=10
    )

    text_config = CLAPTextCfg()
    audio_config = CLAPAudioCfp()

    model = CLAP(
        1234, # ? embed_dim
        audio_config,
        text_config,
        quick_gelu=False
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model.encode_audio, transform
