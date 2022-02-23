from typing import Callable
import numpy as np
import torch

def splice_audio(audio: np.ndarray, transform: Callable, chunk_size: int = 100_000, num_chunks: int = 8) -> torch.Tensor:
    total_size = chunk_size * num_chunks

    if audio.shape[0] > total_size:
        # Truncate audio
        audio = audio[:total_size]
    elif audio.shape[0] < total_size:
        # Pad audio
        pad_size = total_size - audio.shape[0]
        zeros = np.zeros(pad_size, dtype=audio.dtype)
        audio = np.concatenate((audio, zeros), axis=0)
    
    chunks = np.array(np.split(audio, num_chunks, axis=1), dtype=audio.dtype)

    return transform(chunks)