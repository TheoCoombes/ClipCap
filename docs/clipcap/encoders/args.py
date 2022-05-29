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
    encoder.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="[optional] In CLIP, this is the number of tiles to split the image into (e.g. 3x3 = 9). In CLAP, this should be the number of splices the audio recieves before being encoded.",
    )
    encoder.add_argument(
        "--window-overlap-percentage",
        type=float,
        default=0.0,
        help="[optional] If enabled, the percentage each window should overlap into each other. Default 0% / non-overlapping.",
    )

    return parser