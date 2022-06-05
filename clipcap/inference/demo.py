from clipcap.inference.args import add_inference_args
from clipcap.inference.no_beam import generate_no_beam

from clipcap.encoders.base import get_encoder_from_model
from clipcap.model.load import load

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from open_clip import tokenize
import numpy as np
import torch

def inference_demo(args: Namespace) -> int:
    model, tokenizer = load(
        args.model_path,
        args.config_path,
        device=args.device,
        from_checkpoint=args.is_checkpoint
    )

    if args.text_prefix is not None:
        text_prefix = tokenizer.bos_token + args.text_prefix
    else:
        text_prefix = tokenizer.bos_token

    text_prefix_tokens = tokenizer.encode(text_prefix, return_tensors="pt").to(args.device)

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
        desired_sentence_length=args.desired_sentence_length,
    )

    caption_tokens = tokenize(captions).to(args.device)

    with torch.no_grad():
        if model.config.encoder_config.use_windowed_embeddings:
            media_features = media_features[0][0].unsqueeze(0)
        
        text_features = encode_method.model.encode_text(caption_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        media_features /= media_features.norm(dim=-1, keepdim=True)

        media_features_mlp = encode_method.model.audio_transform(media_features)
        text_features_mlp = encode_method.model.text_transform(text_features)

        sim_media_text = media_features.cpu().numpy() @ text_features_mlp.cpu().numpy().T
        sim_mlp_media_text = media_features_mlp.cpu().numpy() @ text_features.cpu().numpy().T

        similarities = (sim_media_text + sim_mlp_media_text) / 2

        mean_similarity = float(np.mean(similarities))
        best_idx = int(np.argmax(similarities))
        similarities = similarities.tolist()

        
    best = captions[best_idx]
    for caption, similarity in zip(captions, similarities):
        print("sim", similarity, "caption", caption)
    print("mean sim", mean_similarity)
    print("best", best)
    
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