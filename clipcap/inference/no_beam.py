from clipcap.model import ClipCapModel, ClipCapModelPrefixOnly

from typing import Union, Callable, Optional
import torch.nn.functional as nnf
import random
import torch


def generate_no_beam(
    model: Union[ClipCapModel, ClipCapModelPrefixOnly],
    tokenizer: Callable,
    embeds: torch.Tensor,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    top_p: float = 0.9,
    top_k: float = 0.0,
    entry_length: int = 67,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    desired_sentence_length: int = 50,
    sentence_length_factor: float = 1.0,
):
    stop_token = tokenizer.encode(tokenizer.eos_token)[0]
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_input_embeddings()(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        embeds_init = embeds
        for top_p in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            for temperature in [0.9, 0.95, 1.0]:
                tokens = text_prefix_tokens
                embeds = embeds_init
                for _ in range(entry_length):
                    # Get logits from a forward pass
                    outputs = model.language_model(inputs_embeds=embeds)
                    logits = outputs.logits

                    # Assume batch size of 1
                    assert logits.shape[0] == 1
                    logits = logits[0, -1, :]

                    # Apply the repetition penalty
                    # if repetition_penalty != 1.0 and tokens is not None:
                    #     tokens1 = tokens[0, :] # assuming batch size of 1
                    #     logits = repetition_penalty_apply(logits, tokens1, repetition_penalty)

                    # Apply temperature and filter
                    logits = logits / (temperature if temperature > 0 else 1.0)
                    logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)

                    # Apply sentence length penalty.
                    if tokens is not None:
                        tokens1 = tokens[0, :] # assuming batch size of 1
                        logits = sentence_length_penalty_apply(
                            logits, tokens1, stop_token, tokens.shape[1],
                            desired_sentence_length, sentence_length_factor
                        )

                    # Get the next token and its embedding
                    probabilities = nnf.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, 1).unsqueeze(0)
                    next_token_embed = model.language_model.get_input_embeddings()(next_token)

                    if stop_token == next_token.item():
                        break
                    else:
                        if tokens is not None:
                            tokens = torch.cat((tokens, next_token), dim=1)
                        else:
                            tokens = next_token
                        
                        embeds = torch.cat((embeds, next_token_embed), dim=1)

                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = tokenizer.decode(output_list)

                print(output_text, "top_p", top_p, "temperature", temperature, "repetition_penalty", repetition_penalty)
            
                generations.append(output_text)
    
    return generations

