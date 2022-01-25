from torchvision.transforms import Compose
import torch.nn.functional as nnf
from clip.model import CLIP
from typing import Union
import skimage.io as io
from PIL import Image
import numpy as np
import torch
import json
import clip
import fire

from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from lms import GPT2, GPT2Tokenizer


def generate_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2Tokenizer,
    embed: torch.Tensor,
    beam_size: int = 5,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    tokens = None
    scores = None

    # .type_as(...) is used to support parallel data in pytorch-lightning
    seq_lengths = torch.ones(beam_size, device="cuda:4")
    has_stopped = torch.zeros(beam_size, dtype=torch.bool, device="cuda:4")

    print(embed.dtype)
    print(seq_lengths.dtype)
    print(has_stopped.dtype)

    with torch.no_grad():
        for _ in range(entry_length):
            outputs = model.language_model.call(inputs_embeds=embed)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                embed = embed.expand(beam_size, *embed.shape[1:])
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

                embed = embed[next_tokens_source]
                scores = scores_sum_average * seq_lengths

                has_stopped = has_stopped[next_tokens_source]
            
            next_token_embed = model.language_model.get_embedding_text(next_tokens.squeeze()).view(embed.shape[0], 1, -1)
            embed = torch.cat((embed, next_token_embed), dim=1)
            has_stopped = has_stopped + next_tokens.eq(stop_token).squeeze()

            if has_stopped.all():
                break
                
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode_tokens(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]

    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts


def generate_no_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2Tokenizer,
    embeds: torch.Tensor,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    tokens = None

    filter_value = -float(np.inf)

    with torch.no_grad():
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
        output_text = tokenizer.decode_tokens(output_list)
    
    return output_text


def demo_generate_caption(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2Tokenizer,
    clip_model: CLIP,
    clip_preproc: Compose,
    image: Image.Image,
    use_beam_search: bool = False,
    device: str = "cuda:0",
    **generation_kwargs
) -> str:
    image = clip_preproc(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
    
    if use_beam_search:
        generated_caption = generate_beam(model, tokenizer, prefix_embed, **generation_kwargs)
    else:
        generated_caption = generate_no_beam(model, tokenizer, prefix_embed, **generation_kwargs)
    
    return generated_caption


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
        if not load_full_model:
            language_model = GPT2.create(language_model_variant)
        tokenizer = GPT2Tokenizer.create(language_model_variant)
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

            caption = demo_generate_caption(
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
    device: str = "cuda:0",
    use_beam_search: bool = True,
    out_filename_prefix: str = "demo_inference",
    **kwargs
):
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    lm = GPT2.create("gpt2-xl")
    tokenizer = GPT2Tokenizer.create("gpt2-xl")

    model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=lm, **kwargs)
    model = model.to(device)
    model = model.eval()

    from pathlib import Path
    from tqdm import tqdm
    import json

    samples_path = Path(shutterstock_path)
    sample_data = {}

    for image_file in tqdm(samples_path.glob("*.jpg"), desc='inference'):
        try:
            image = io.imread(image_file)
            image = Image.fromarray(image)

            metadata_file = image_file.parent / image_file.name.replace(".jpg", ".json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            caption= demo_generate_caption(
                model, tokenizer, clip_model, preprocess, image,
                use_beam_search=use_beam_search, device=device
            )

            url = metadata["src"]
            original_caption = metadata["alt"]

            text_inputs = torch.cat([
                clip.tokenize(caption, truncate=True),
                clip.tokenize(original_caption, truncate=True)
            ]).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                text_features = clip_model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = image_features.cpu().numpy() @ text_features.cpu().numpy().T

            print(similarities)
            generated_sim, original_sim = similarities
            

            sample_data[url] = {
                "original_caption": original_caption,
                "original_sim": float(original_sim[0]),
                "generated_caption": caption,
                "generated_sim": float(generated_sim[0])
            }

        except Exception as e:
            print("exception:", e)
            break
    
    with open(f"{out_filename_prefix}_shutterstock.json", "w+") as f:
        json.dump(sample_data, f)


if __name__ == "__main__":
    fire.Fire(_shutterstock_demo)