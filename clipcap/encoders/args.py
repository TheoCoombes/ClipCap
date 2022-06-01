from argparse import ArgumentParser

def add_encoder_args(parser: ArgumentParser) -> ArgumentParser:
    encoder = parser.add_argument_group('encoder')
    encoder.add_argument(
        "--encoder-model-name",
        choices=["clip", "clap"],
        type=str,
        default="clip",
        help="Name of encoder model ('clip' or 'clap').",
    )
    encoder.add_argument(
        "--encoder-model-variant",
        type=str,
        default="ViT-L_14",
        help="The specific version of CLIP e.g. 'ViT-L_14' ('_' gets replaced with a forward slash), or the path to the CLAP checkpoint.",
    )
    encoder.add_argument(
        "--normalize-embeddings",
        type=bool,
        default=False,
        help="Whether or not the generated embeddings should be normalized.",
    )

    windowed = parser.add_argument_group('windowed')
    windowed.add_argument(
        "--use-windowed-embeddings",
        type=bool,
        default=False,
        help="Transforms the data into 'windows' so that more embeddings can be generated without needing fine-grained models.",
    )
    windowed.add_argument(
        "--window-size",
        type=int,
        default=(4 * 4),
        help="If enabled: In CLIP, this is the number of tiles to split the image into (e.g. 3x3 = 9). In CLAP, this should be the number of splices the audio recieves before being encoded.",
    )
    windowed.add_argument(
        "--window-overlap-percentage",
        type=float,
        default=0.0,
        help="If enabled, the percentage each window should overlap into each other. Default no overlap.",
    )

    return parser