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
        default=8,
        help="Number of attention heads in the mapping transformer.",
    )
    model.add_argument(
        "--use-positional-embeddings",
        type=bool,
        default=True,
        help="If windowed embeddings were enabled in preprocessing, use positional embeddings for windowed sequence in the mapping transformer.",
    )

    return parser