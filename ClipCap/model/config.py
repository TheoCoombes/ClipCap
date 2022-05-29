from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    optimizer_lr: float = 2e-5
    use_deepspeed_optimisers: bool = True
    scheduler_warmup_steps: int = 123
    total_steps: int = 123

def training_config_from_args(args: ArgumentParser) -> dict:
    return TrainingConfig(
        optimizer_lr=args.optimizer_lr,
        use_deepspeed_optimisers=args.deepspeed,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        total_steps=args.total_steps
    )


@dataclass
class Config:
    language_model: str = "gpt2-xl"
    prefix_length: int = 10
    projection_length: int = 10
    transformer_layers: int = 8
    transformer_attention_heads: int = 16

    use_windowed_embeddings: bool = False
    window_size: Optional[int] = None # (4 * 4)
    window_overlap_percentage: float = 0.0
    use_positional_embeddings: bool = True

    training_config: Optional[TrainingConfig] = None

def model_config_from_args(args: ArgumentParser) -> dict:
    return Config(
        language_model = args.language_model,
        prefix_length = args.prefix_length,
        projection_length = args.projection_length,
        transformer_layers = args.transformer_layers,
        transformer_attention_heads = args.transformer_attention_heads,
        use_windowed_embeddings = args.use_windowed_embeddings,
        window_size = args.window_size,
        use_positional_embeddings = args.use_positional_embeddings,
        training_config = None
    )


# TODO implement base model configs for easy reproduction?