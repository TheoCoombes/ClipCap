from dataclasses import dataclass, asdict
from argparse import Namespace
from typing import Optional

@dataclass
class EncoderConfig:
    encoder_model_name: str = "clip"
    encoder_model_variant: str = "ViT-L/14"
    encoder_embedding_size: Optional[int] = None # Calculated during dataloading.
    normalize_embeddings: bool = False

    use_windowed_embeddings: bool = False
    window_size: int = (4 * 4)
    window_overlap_percentage: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_args(cls: "EncoderConfig", args: Namespace) -> "EncoderConfig":
        return cls(
            encoder_model_name=args.encoder_model_name,
            encoder_model_variant=args.encoder_model_variant,
            encoder_embedding_size=None,
            normalize_embeddings=args.normalize_embeddings,
            use_windowed_embeddings=args.use_windowed_embeddings,
            window_size=args.window_size,
            window_overlap_percentage=args.window_overlap_percentage
        )