"""mapper module transform images and text to embeddings"""

import torch
from torch.nn import Module


class EncoderMapper:
    """transforms media and texts into embeddings"""

    def __init__(self, model: Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def __call__(self, item):
        with torch.no_grad():
            features = self.model(item["data_tensor"].to(self.device))
            
            embeddings = features.cpu().numpy()
            text = item["text"]

            return {
                "embeddings": embeddings,
                "text": text,
            }
