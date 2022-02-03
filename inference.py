from torchvision.transforms import Compose
from typing import Tuple, List, Optional
import torch.nn.functional as nnf
from clip.model import CLIP
from typing import Union
import skimage.io as io
from PIL import Image
import numpy as np
import torch
import clip
import fire

from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from lms import (
    GPT2, GPT2_Tokenizer,
    GPTJ, GPTJ_Tokenizer,
    T0, T0_Tokenizer
)


def generate_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    beam_size: int = 5,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    tokens = None
    scores = None

    seq_lengths = torch.ones(beam_size, device=embeds.device)
    has_stopped = torch.zeros(beam_size, dtype=torch.bool, device=embeds.device)

    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        for i in range(number_to_generate):
            for _ in range(entry_length):
                outputs = model.language_model.call(inputs_embeds=embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()

                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    embeds = embeds.expand(beam_size, *embeds.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[has_stopped] = -float(np.inf)
                    logits[has_stopped, 0] = 0

                    scores_sum = scores[:, None] + logits
                    seq_lengths[~has_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)

                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)

                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)

                    embeds = embeds[next_tokens_source]
                    scores = scores_sum_average * seq_lengths

                    has_stopped = has_stopped[next_tokens_source]
                
                next_token_embed = model.language_model.get_embedding_text(next_tokens.squeeze()).view(embeds.shape[0], 1, -1)
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                has_stopped = has_stopped + next_tokens.eq(stop_token).squeeze()

                if has_stopped.all():
                    break
                
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode_tokens(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]

            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order][0]

            generations.append(output_texts)
    
    return generations


def generate_no_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    tokens = None

    filter_value = -float(np.inf)
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        for i in range(number_to_generate):
            for _ in range(entry_length):
                outputs = model.language_model.call(inputs_embeds=embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.language_model.get_embedding_text(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                
                if stop_token == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode_tokens(output_list)[0]
        
            generations.append(output_text)
    
    return generations


def demo_generate_captions(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    clip_model: CLIP,
    clip_preproc: Compose,
    image: Image.Image,
    number_to_generate: int = 1,
    text_prefix: Optional[str] = None,
    use_beam_search: bool = False,
    device: str = "cuda:0",
    **generation_kwargs
) -> Tuple[List[str], torch.Tensor]:
    image = clip_preproc(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
    
    if text_prefix is not None:
        text_prefix_tokens = torch.tensor(
            tokenizer.encode_text(text_prefix), device=device
        ).unsqueeze(0)
    else:
        text_prefix_tokens = None
    
    if use_beam_search:
        generated_captions = generate_beam(model, tokenizer, prefix_embed,
            number_to_generate=number_to_generate, text_prefix_tokens=text_prefix_tokens,
            **generation_kwargs)
    else:
        generated_captions = generate_no_beam(model, tokenizer, prefix_embed,
            number_to_generate=number_to_generate, text_prefix_tokens=text_prefix_tokens,
            **generation_kwargs)
    
    if text_prefix is not None:
        generated_captions = [(text_prefix + caption) for caption in generated_captions]
    
    return generated_captions, prefix


def demo(
    checkpoint_path: str = "./train/latest.pt",
    prefix_length: int = 40,
    clip_prefix_length: int = 40,
    prefix_size: int = 512,
    mapping_type: str = "mlp",
    num_layers: int = 8,
    only_prefix: bool = False,
    language_model_type: str = "gpt2",
    language_model_variant: str = "gpt2-xl",
    clip_model_type: str = "ViT-B/32",
    load_full_model: bool = True,
    use_beam_search: bool = False,
    device: str = "cuda:0",
    **generation_kwargs
):
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    if language_model_type == "gpt2":
        language_model = GPT2.create(language_model_variant)
        tokenizer = GPT2_Tokenizer.create(language_model_variant)
    else:
        raise ValueError(f"invalid language model type: '{language_model_type}' (expected 'gpt2')")

    if only_prefix:
        if load_full_model:
            model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path)
        else:
            model = CLIPCaptionPrefixOnly(
                language_model, prefix_length, clip_length=clip_prefix_length,
                prefix_size=prefix_size, num_layers=num_layers, mapping_type=mapping_type
            )
            model.load_state_dict(torch.load(checkpoint_path))
    else:
        if load_full_model:
            model = CLIPCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
        else:
            model = CLIPCaptionModel(
                language_model, prefix_length, clip_length=clip_prefix_length,
                prefix_size=prefix_size, num_layers=num_layers, mapping_type=mapping_type
            )
            model.load_state_dict(torch.load(checkpoint_path))
    
    model = model.to(device)
    model = model.eval()

    try:
        while True:
            print("CLIP-Image-Captioning inference demo\n")

            image_path_url = input("enter image url or path > ")

            image = io.imread(image_path_url)
            image = Image.fromarray(image)

            caption = demo_generate_captions(
                model, tokenizer, clip_model, preprocess, image,
                use_beam_search=use_beam_search, device=device, **generation_kwargs
            )

            print(caption)
            print()
    except KeyboardInterrupt:
        print("exiting...")
        exit(0)


def _shutterstock_demo(
    checkpoint_path: str,
    shutterstock_path: str,
    number_to_generate: int = 1,
    text_prefix: Optional[str] = None,
    device: str = "cuda:0",
    use_beam_search: bool = True,
    prefix_only: bool = False,
    out_filename_prefix: str = "demo_inference",
    clip_model: str = "ViT-B/32",
    language_model_type: str = "gpt2",
    language_model_variant: str = "gpt2-xl",
    hf_cache_dir: Optional[str] = None,
    total_samples: int = 100,
    load_pl_checkpoint: bool = True,
    **model_kwargs
):
    clip_model, preprocess = clip.load(clip_model, device=device, jit=False)

    if language_model_type == "gpt2":
        language_model = GPT2.create(language_model_variant, cache_dir=hf_cache_dir)
        tokenizer = GPT2_Tokenizer.create(language_model_variant, cache_dir=hf_cache_dir)
    elif language_model_type in ("gptj", "gpt-j"):
        language_model = GPTJ.create(language_model_variant, cache_dir=hf_cache_dir)
        tokenizer = GPTJ_Tokenizer.create(language_model_variant, cache_dir=hf_cache_dir)
    elif language_model_type in ("t0", "T5"):
        language_model = T0.create(language_model_variant, cache_dir=hf_cache_dir)
        tokenizer = T0_Tokenizer.create(language_model_variant, cache_dir=hf_cache_dir)
    else:
        raise ValueError(f"invalid language model type '{language_model_type}' (expected 'gpt-j' / 'gpt2' / 't0' / 't5')")

    if load_pl_checkpoint:
        if prefix_only:
            model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model)
        else:
            model = CLIPCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model)
    else:
        if prefix_only:
            model = CLIPCaptionPrefixOnly(language_model, **model_kwargs)
        else:
            model = CLIPCaptionModel(language_model, **model_kwargs)
        
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(checkpoint_path))
    
    model = model.to(device)
    model = model.eval()

    from pathlib import Path
    from tqdm import tqdm
    import json

    samples_path = Path(shutterstock_path)
    sample_data = {}

    for image_file in tqdm(sorted(list(samples_path.glob("*.jpg")), key=lambda x: x.name)[:total_samples], desc='inference'):
        image = io.imread(image_file)
        pil_image = Image.fromarray(image)

        metadata_file = image_file.parent / image_file.name.replace(".jpg", ".json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        captions, image_features = demo_generate_captions(
            model, tokenizer, clip_model, preprocess, pil_image,
            use_beam_search=use_beam_search, device=device,
            number_to_generate=number_to_generate, text_prefix=text_prefix
        )

        url = metadata["src"]
        original_caption = metadata["alt"]

        text_inputs = clip.tokenize([original_caption, *captions], truncate=True).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = image_features.cpu().numpy() @ text_features.cpu().numpy().T
        similarities = similarities.tolist()

        original_sim = similarities[0][0]
        generated_sims = similarities[0][1:]

        best_sim = max(generated_sims)
        best_caption = captions[generated_sims.index(best_sim)]

        sample_data[url] = {
            "original_caption": original_caption,
            "original_sim": original_sim,
            "generated_captions": captions,
            "generated_sim": generated_sims,
            "best_caption": best_caption,
            "best_sim": best_sim
        }
    
    with open(f"{out_filename_prefix}_shutterstock.json", "w+") as f:
        json.dump(sample_data, f)


if __name__ == "__main__":
    fire.Fire(_shutterstock_demo)
