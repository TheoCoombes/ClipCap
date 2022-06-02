from clipcap.inference.args import add_inference_args
from clipcap.inference.base import *

from clipcap.encoders.base import get_encoder_from_model
from clipcap.model.load import load

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from open_clip import tokenize
import torch

def inference_demo(args: Namespace) -> int:
    model, tokenizer = load(
        args.model_path,
        args.config_path,
        device=args.device,
        from_checkpoint=args.is_checkpoint
    )

    if args.text_prefix is not None:
        text_prefix_tokens = tokenizer.encode(args.text_prefix, return_tensors="pt").to(args.device)
    else:
        text_prefix_tokens = None

    encode_method, sample_processor = get_encoder_from_model(model, device=args.device)

    sample = sample_processor(args.sample_path).unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        media_features = encode_method(sample)
        prefix = model.transformer_mapper(media_features)

    captions = generate_no_beam(
        model, tokenizer, prefix,
        number_to_generate=args.number_to_generate,
        text_prefix_tokens=text_prefix_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        desired_sentence_length=args.desired_sentence_length
    )


    caption_tokens = tokenize(captions).to(args.device)

    with torch.no_grad():
        text_features = encode_method.model.encode_text(caption_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (media_features @ text_features.T)
        mean_similarity = torch.mean(similarities)
        _, indices = similarities.softmax(dim=-1)[0].topk(1)

    caption_idx = indices[0]
    caption = captions[caption_idx]

    print(captions)
    print("mean sim", mean_similarity)
    print("best", caption)
    
    return 0


def run_inference_demo() -> int:
    """
    Main inference function.
    """
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser = add_inference_args(parser)
    args = parser.parse_args()
    return inference_demo(args)

if __name__ == '__main__':
    exit(run_inference_demo())