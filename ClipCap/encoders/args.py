from argparse import ArgumentParser

def add_encoder_args(parser: ArgumentParser) -> ArgumentParser:
    encoder = parser.add_argument_group('encoder')

    encoder.add_argument(
        "--encoder-model-name",
        choices=["clip", "clap"],
        type=str,
        default="clip",
        help="Name of encoder model ('clip' / 'clap').",
    )
    encoder.add_argument(
        "--encoder-model-variant",
        type=str,
        default="ViT-L/14",
        help="The specific version of CLIP e.g. 'ViT-L/14', or the path to the CLAP checkpoint.",
    )

    return parser