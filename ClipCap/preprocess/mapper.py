"""mapper module transform images and text to embeddings"""

import torch
from torch.nn import Module


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class EncoderMapper:
    """transforms media and texts into embeddings"""

    def __init__(self, model: Module, normalize: bool = True, device: str = "cuda"):
        self.model = model
        self.normalize = normalize
        self.device = device

    def __call__(self, item):
        with torch.no_grad():
            features = self.model(item["data_tensor"].to(self.device))
            if self.normalize: features /= features.norm(dim=-1, keepdim=True)
            embeddings = features.cpu().numpy()
            
            text = item["text"]

            return {
                "embeddings": embeddings,
                "text": text,
            }
