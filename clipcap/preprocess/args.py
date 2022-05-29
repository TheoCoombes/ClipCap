from argparse import ArgumentParser

def add_preprocess_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=None,
        help="path to the training dataset (local or remote).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./train/",
        help="output path to store the preprocessed data.",
    )
    parser.add_argument(
        "--input-format",
        choices=["files", "webdataset"],
        type=str,
        default="files",
        help="type of dataset: either a `files` for a folder containing media and .txt files, or `webdataset`",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="number of samples to process in each batch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to generate embeds on (default: cuda:0)",
    )
    parser.add_argument(
        "--distribution-strategy",
        choices=["sequential", "pyspark"],
        type=str,
        default="sequential",
        help="see rom1504/clip-retrieval readme for more (pyspark is untested)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of workers to use for the dataloader",
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=bool,
        default=True,
        help="whether or not the generated embeddings should be normalized",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10**6,
        help="max number of processed samples to store at once",
    )
    parser.add_argument(
        "--output-partition-count",
        type=int,
        default=None,
        help="number of output partitions",
    )
    
    wds = parser.add_argument_group('--input-format=webdataset')
    wds.add_argument(
        "--wds-media-key",
        type=str,
        default="jpg",
        help="[webdataset reader only] key to use for the content to be embedded",
    )
    wds.add_argument(
        "--wds-caption-key",
        type=str,
        default="txt",
        help="[webdataset reader only] key to use for the captions",
    )
    wds.add_argument(
        "--wds-samples-per-file",
        type=int,
        default=10_000,
        help="[webdataset reader only] the number of samples stored in each .tar",
    )
    wds.add_argument(
        "--wds-cache-path",
        type=str,
        default=None,
        help="[webdataset reader only] webdataset cache path (optional)",
    )

    files = parser.add_argument_group('--input-format=files')
    files.add_argument(
        "--media-file-extensions",
        type=str,
        default="png,jpg,jpeg,bmp",
        help="[files reader only] comma seperated file extensions for the media to be loaded, e.g. 'mp3,wav'.",
    )

    return parser