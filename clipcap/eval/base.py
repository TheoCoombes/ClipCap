from clipcap.model.model import ClipCapModel, ClipCapModelPrefixOnly
from clipcap.encoders.base import get_encoder_from_model
from clipcap.inference.args import add_inference_args
from clipcap.inference.base import generate_beam
from clipcap.model.load import load

from clipcap.eval.dataset import EvalDataset
from clipcap.eval.metrics import evaluate_metrics

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Tuple, Union, Callable
from braceexpand import braceexpand
from pprint import pprint
import pandas as pd
import torch
import json

_DEFAULT_REFERENCE_CAPTION_COLUMNS = "caption_reference_{{00..{:02d}}}"
_DEFAULT_PREDICTION_COLUMN = "caption_predicted"
_DEFAULT_FILENAME_COLUMN = "file_name"

def prepare_model(args: Namespace) -> Tuple[Union[ClipCapModel, ClipCapModelPrefixOnly], Callable, Callable, Callable]:
    model, tokenizer = load(
        args.model_path,
        args.config_path,
        device=args.device,
        from_checkpoint=args.is_checkpoint
    )

    encode_method, sample_processor = get_encoder_from_model(model, device=args.device)

    return model, tokenizer, encode_method, sample_processor

def eval(args: Namespace) -> int:
    model, tokenizer, encode_method, sample_processor = prepare_model(args)

    dataset = EvalDataset(sample_processor, args.sample_path)
    predictions = []

    for filename, sample in dataset:
        sample = sample.unsqueeze(0).to(args.device)
    
        with torch.no_grad():
            embeds = encode_method(sample)
            prefix = model.transformer_mapper(embeds)

        captions = generate_beam(
            model, tokenizer, prefix,
            number_to_generate=1
        )

        caption = captions[0]
        
        predictions.append({
            _DEFAULT_FILENAME_COLUMN: filename,
            _DEFAULT_PREDICTION_COLUMN: caption
        })
    
    col_list = [args.csv_filename_column, *braceexpand(args.csv_reference_caption_columns)]
    reference = pd.read_csv(args.reference_csv, usecols=col_list)

    number_reference_captions = len(col_list) - 1
    new_brace = _DEFAULT_REFERENCE_CAPTION_COLUMNS.format(number_reference_captions)
    new_col_list = [_DEFAULT_FILENAME_COLUMN, *braceexpand(new_brace)]

    reference = reference.rename(columns={old: new for old, new in zip(col_list, new_col_list)})
    reference = reference.to_dict('records')

    scores = evaluate_metrics(
        predictions, reference, nb_reference_captions=number_reference_captions
    )

    if args.save_file is not None:
        with open(args.save_file, "w+") as f:
            json.dump(scores, f)
        
    pprint(scores)

    return 0


def run_eval() -> int:
    """
    Main inference function.
    """
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser = add_inference_args(parser)
    args = parser.parse_args()
    return eval(args)


if __name__ == '__main__':
    exit(run_eval())