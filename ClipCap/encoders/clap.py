from typing import Tuple, Callable, Optional
from torch.nn import functional as nnf
from torch.nn import Module
from io import BytesIO
import torch

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


def get_clap_encoder(model_path: str, window_size: Optional[int] = None,
                     window_overlap_percentage: float = 0.0, device: str = "cuda") -> Tuple[Module, Callable]:
    import open_clip # where open_clip = the LAION-AI/CLAP fork
    from collections import OrderedDict
    
    transform = CLAPTransform(
        sample_rate=48000,
        mono=True,
        max_duration=10
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

    encode_audio = lambda x: model.encode_audio(x)["embedding"]

    return encode_audio, transform