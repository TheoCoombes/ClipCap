from argparse import ArgumentParser

def add_inference_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-path",
        type=str,
        default="./model.ckpt",
        help="Path to the model, either the checkpoint (`.ckpt`) or state_dict (`.pt`).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./model_config.yaml",
        help="Path to the config yaml file created by the training script (usually in same directory as the model).",
    )
    parser.add_argument(
        "--is-checkpoint",
        type=bool,
        default=False,
        help="If the file at `--model-path` is a training checkpoint (`.ckpt`) or not.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load the model and encoder onto.",
    )

    inference = parser.add_argument_group('inference')
    inference.add_argument(
        "--sample-path",
        type=str,
        default="./image.jpg",
        help="Path to the sample used for inference. In eval, this is the directory containing eval samples with the corresponding filenames to the csv.",
    )
    inference.add_argument(
        "--number-to-generate",
        type=int,
        default=5,
        help="Number of captions to be generated.",
    )
    inference.add_argument(
        "--text-prefix",
        type=str,
        default=None,
        help="Add a textual prefix to the generated captions (useful for VQA tasks), i.e. 'Q: What is the man doing? A:'.",
    )
    inference.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Inference settings: top_p.",
    )
    inference.add_argument(
        "--top-k",
        type=float,
        default=0.0,
        help="Inference settings: top_k.",
    )
    inference.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Inference settings: temperature.",
    )
    inference.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Inference settings: sentence repeititon penalty.",
    )
    inference.add_argument(
        "--desired-sentence-length",
        type=int,
        default=77,
        help="Inference settings: desired sentence length.",
    )
    

    return parser