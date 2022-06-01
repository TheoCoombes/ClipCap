from clipcap.encoders.config import EncoderConfig

from dataclasses import dataclass, asdict
from argparse import Namespace
from typing import Optional

@dataclass
class TrainingConfig:
    optimizer_lr: float = 2e-5
    use_deepspeed_optimisers: bool = True
    scheduler_warmup_steps: int = 123
    total_steps: int = 123

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_args(cls: "TrainingConfig", args: Namespace) -> "TrainingConfig":
        return cls(
            optimizer_lr=args.optimizer_lr,
            use_deepspeed_optimisers=args.enable_deepspeed,
            scheduler_warmup_steps=args.scheduler_warmup_steps,
            total_steps=args.total_steps
        )


@dataclass
class Config:
    language_model: str = "gpt2-xl"
    train_language_model: bool = False
    prefix_length: int = 10
    projection_length: int = 10
    transformer_layers: int = 8
    transformer_attention_heads: int = 16
    use_positional_embeddings: bool = True

    encoder_config: Optional[EncoderConfig] = None
    training_config: Optional[TrainingConfig] = None

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_args(cls: "Config", args: Namespace) -> "Config":
        return cls(
            language_model = args.language_model,
            train_language_model = args.train_language_model,
            prefix_length = args.prefix_length,
            projection_length = args.projection_length,
            transformer_layers = args.transformer_layers,
            transformer_attention_heads = args.transformer_attention_heads,
            use_positional_embeddings = args.use_positional_embeddings,
            encoder_config = None,
            training_config = None
        )


# TODO implement base model configs for easy reproduction?