from typing import Tuple, Callable
from io import BytesIO
import numpy as np
import torch

def _Transform(file: BytesIO) -> torch.Tensor:
    import soundfile as sf
    import scipy.signal as sps

    audio_data, sample_rate = sf.read(file)
    
    if sample_rate != 48000:
        number_of_samples = round(len(audio_data) * float(48000) / sample_rate)
        audio_data = sps.resample(audio_data, number_of_samples)

    if len(audio_data) > 480000:
        audio_data = audio_data[:480000]
    elif len(audio_data) < 480000:
        audio_data = np.pad(
            audio_data,
            (0, 480000 - len(audio_data)),
            mode="constant",
            constant_values=0,
        )
    
    waveform = torch.tensor(audio_data).float()
    
    return waveform


def get_clap_encoder(model_path: str, device: str = "cuda") -> Tuple[torch.Module, Callable]:
    from open_clip import CLAP # where open_clip = the LAION-AI/CLAP fork

    