from argparse import ArgumentParser

def add_eval_args(parser: ArgumentParser) -> ArgumentParser:
    eval = parser.add_argument_group('eval')
    eval.add_argument(
        "--reference-csv",
        type=str,
        default="./eval.csv",
        help="Path to csv filled with reference captions. See `mode` for dataset specific csv specifications.",
    )
    eval.add_argument(
        "--csv-filename-column",
        type=str,
        default="file_name",
        help="Column containing filenames in the csv.",
    )
    eval.add_argument(
        "--csv-reference-caption-columns",
        type=str,
        default="caption_reference_{00..05}",
        help="Column(s) containing the ground truth captions (brace expandable).",
    )
    eval.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="Path to json file to dump eval metrics to (optional).",
    )
    

    return parser