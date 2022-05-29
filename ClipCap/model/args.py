from argparse import ArgumentParser

def add_model_args(parser: ArgumentParser) -> ArgumentParser:
    model = parser.add_argument_group('model')
    model.add_argument(
        "--language-model",
        type=str,
        default="gpt2-xl",
        help="Frozen language model loaded using the transformers.AutoModelForCausalLM class.",
    )
    model.add_argument(
        "--prefix-length",
        type=int,
        default=10,
        help="Length in text (LM) embeddings of the prefix placed after the embeddings.",
    )
    model.add_argument(
        "--projection-length",
        type=int,
        default=10,
        help="The number of LM embeddings a single media (e.g. CLIP) embedding should be projected into.",
    )
    model.add_argument(
        "--train-language-model",
        type=bool,
        default=False,
        help="Whether or not the language model should remain unfrozen during training.",
    )
    model.add_argument(
        "--transformer-layers",
        type=int,
        default=8,
        help="Number of layers in the mapping transformer.",
    )
    model.add_argument(
        "--transformer-attention-heads",
        type=int,
        default=16,
        help="Number of attention heads in the mapping transformer.",
    )

    windowed = parser.add_argument_group('windowed')
    windowed.add_argument(
        "--use-windowed-embeddings",
        type=bool,
        default=False,
        help="Whether or not to use multiple encoder outputs / full ViT. For CLIP, this splits the image into a sequence of tiled embeddings. For CLAP, this splits the audio into a fixed amount of segments.",
    )
    windowed.add_argument(
        "--window-size",
        type=int,
        default=None, # (4 * 4)
        help="[optional] This should be the same as used in the preprocessor. This is equal to `tensor.shape[1]` of the preprocessed dataset.",
    )
    windowed.add_argument(
        "--window-overlap-percentage",
        type=float,
        default=0.0,
        help="[optional] If enabled, the percentage each window should overlap into each other. Default 0% / non-overlapping.",
    )
    windowed.add_argument(
        "--use-positional-embeddings",
        type=bool,
        default=True,
        help="Use positional embeddings for windowed sequence in the mapping transformer.",
    )

    return parser