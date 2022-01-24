import torch.nn.functional as nnf
from typing import Union
import skimage.io as io
from PIL import Image
import numpy as np
import torch
import json
import clip

from model import CLIPCaptionModel, CLIPCaptionPrefix
from lms import GPT2Tokenizer


def generate_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefix],
    tokenizer: GPT2Tokenizer,
    embed: torch.Tensor,
    beam_size: int = 5,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_tokens(stop_token)[0]
    tokens = None
    scores = None

    # .type_as(...) is used to support parallel data in pytorch-lightning
    seq_lengths = torch.ones(beam_size).type_as(embed)
    has_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool).type_as(embed)

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
            
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(embed.shape[0], 1, -1)
            embed = torch.cat((embed, next_token_embed), dim=1)
            has_stopped = has_stopped + next_tokens.eq(stop_token).squeeze()

            if has_stopped.all():
                break
                
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]

    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts


def generate_no_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefix],
    tokenizer: GPT2Tokenizer,
    embeds: torch.Tensor,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_tokens(stop_token)[0]
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
            next_token_embed = model.gpt.transformer.wte(next_token)

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


device = "cuda:7"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", cache_dir='/mnt/theocoombes/huggingface-cache')



prefix_length = 40
use_beam_search = True

model = ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=512,
                                  num_layers=8, mapping_type='transformer')
model.load_state_dict(torch.load(model_path, map_location=CPU))

model = model.eval()
model = model.to(device)


# ----------
print("\n\n")

samples_path = Path("./test/image-photo/")
sample_data = {}

for image_file in samples_path.glob("*.jpg"):
    try:
        image = io.imread(image_file)
        image = Image.fromarray(image)

        metadata_file = image_file.parent / image_file.name.replace(".jpg", ".json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)

            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            

        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        url = metadata["src"]
        original_caption = metadata["alt"]

        text_inputs = torch.cat([
            clip.tokenize(generated_text_prefix, truncate=True),
            clip.tokenize(original_caption, truncate=True)
        ]).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)

        prefix /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = image_features.cpu().numpy() @ text_features.cpu().numpy().T

        print(similarities)
        break

    except Exception as e:
        print("exception:", e)
        break
