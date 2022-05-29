from argparse import ArgumentParser

def add_training_args(parser: ArgumentParser) -> ArgumentParser:
    training = parser.add_argument_group('training')
    training.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training cycles of the training data before exiting.",
    )
    training.add_argument(
        "--optimizer-lr",
        type=float,
        default=2e-5,
        help="Optimizer loss rate.",
    )
    training.add_argument(
        "--scheduler-warmup-steps",
        type=int,
        default=5000,
        help="LR scheduler warmup duration in steps.",
    )
    training.add_argument(
        "--fp-precision",
        type=int,
        default=32,
        help="Floating point precision. (16/32/64)",
    )
    training.add_argument(
        "--checkpoint-save-frequency",
        type=int,
        default=1,
        help="Save a new checkpoint every 'n' epochs.",
    )
    training.add_argument(
        "--checkpoint-filename-prefix",
        type=str,
        default=1,
        help="Save checkpoint filename with prefix at the beginning.",
    )
    training.add_argument(
        "--device",
        type=str,
        default="0",
        help="Sets the training GPU device(s) to use. '<number>' / '<number>,<number>,...' / '-1' for all",
    )

    data = parser.add_argument_group('data')
    data.add_argument(
        "--input-dataset",
        type=str,
        default="./dataset/",
        help="Path to the preprocessed dataset.",
    )
    data.add_argument(
        "--output-folder",
        type=str,
        default="./models/",
        help="Directory to save trained checkpoints to.",
    )
    data.add_argument(
        "--reader-max-piece-size",
        type=int,
        default=50,
        help="Maximum size of a piece from Rom's EmbeddingReader class. The default value works for most cases. Increase or decrease based on your file system performances (default 50MB)",
    )
    data.add_argument(
        "--reader-parallel-pieces",
        type=int,
        default=10,
        help="Number of pieces to read in parallel from Rom's EmbeddingReader class. Increase or decrease depending on your filesystem.",
    )

    deepspeed = parser.add_argument_group('deepspeed')
    deepspeed.add_argument(
        "--deepspeed",
        type=bool,
        default=False,
        help="Train using deepspeed (required if gpus > 1).",
    )
    deepspeed.add_argument(
        "--deepspeed-strategy",
        type=str,
        default=None,
        help="Deepspeed ZeRo stage. (see https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed)",
    )

    wandb = parser.add_argument_group('wandb')
    wandb.add_argument(
        "--enable-wandb",
        type=bool,
        default=False,
        help="Enable logging stats to wandb.",
    )
    wandb.add_argument(
        "--wandb-project",
        type=str,
        default="clipcap",
        help="The name of the Wandb project.",
    )
    wandb.add_argument(
        "--logging-frequency",
        type=int,
        default=50,
        help="New data is logged every 'n' steps.",
    )

    return parser


# optmizer_lr: float = 2e-5,
# use_deepspeed_optims: bool = True,
# scheduler_warmup_steps: int = 123,
# total_steps: int = 123