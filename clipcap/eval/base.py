from clipcap.model.model import ClipCapModel, ClipCapModelPrefixOnly
from clipcap.encoders.base import get_encoder_from_model
from clipcap.inference.args import add_inference_args
from clipcap.inference.nucleus_sampling import generate_nucleus_sampling
from clipcap.model.load import load

from clipcap.eval.args import add_eval_args
from clipcap.eval.dataset import EvalDataset
from clipcap.eval.metrics import evaluate_metrics_from_lists

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Tuple, Union, Callable
from braceexpand import braceexpand
from open_clip import tokenize
from pprint import pprint
import pandas as pd
import numpy as np
import torch
import json
import tqdm

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
    # model, tokenizer, encode_method, sample_processor = prepare_model(args)

    # dataset = EvalDataset(sample_processor, args.sample_path)
    # predictions = {}

    # for filename, sample in tqdm.tqdm(dataset, desc="eval"):
    #     sample = sample.unsqueeze(0).to(args.device)
    
    #     with torch.no_grad():
    #         media_features = encode_method(sample)
    #         prefix = model.transformer_mapper(media_features)

    #     captions = generate_nucleus_sampling(
    #         model, tokenizer, prefix,
    #         number_to_generate=args.number_to_generate,
    #         text_prefix_tokens=None,
    #         top_p=args.top_p,
    #         top_k=args.top_k,
    #         temperature=args.temperature,
    #         # repetition_penalty=args.repetition_penalty,
    #         # desired_sentence_length=args.desired_sentence_length,
    #     )

    #     caption_tokens = tokenize(captions).to(args.device)

    #     with torch.no_grad():
    #         if model.config.encoder_config.use_windowed_embeddings:
    #             media_features = media_features[0][0].unsqueeze(0)
            
    #         media_features, text_features, media_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = encode_method.model(sample, caption_tokens)
            
    #         text_features /= text_features.norm(dim=-1, keepdim=True)
    #         media_features /= media_features.norm(dim=-1, keepdim=True)

    #         a_logits_per_audio = (logit_scale_a * media_features @ text_features_mlp.t()).detach().cpu()
    #         t_logits_per_audio = (logit_scale_t * media_features_mlp @ text_features.t()).detach().cpu()

    #         similarities = ((a_logits_per_audio + t_logits_per_audio) / 2)[0].numpy()
        
    #     best_idx = int(np.argmax(similarities))
    #     caption = captions[best_idx]
        
    #     predictions[filename] = caption
    
    with open("eval2.json", "r") as f:
        predictions = json.load(f)
    
    reference_df = pd.read_csv(args.reference_csv)
    predictions_list = []
    references_list = []
    ids = []

    for index, row in reference_df.iterrows():
        filename = row[args.csv_filename_column]
        references = [row[key] for key in braceexpand(args.csv_reference_caption_columns)]
        references_list.append(references)
        predictions_list.append(predictions[filename])
        ids.append(index)
    
    print(predictions_list[0])
    print(references_list[0])

    scores = evaluate_metrics_from_lists(
        predictions_list, references_list
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
    parser = add_eval_args(parser)
    parser = add_inference_args(parser)
    args = parser.parse_args()
    return eval(args)


if __name__ == '__main__':
    exit(run_eval())