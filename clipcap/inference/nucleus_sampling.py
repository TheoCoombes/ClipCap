from clipcap.model import ClipCapModel, ClipCapModelPrefixOnly

from clipcap.inference.utils import top_k_top_p_filtering, sentence_length_penalty_apply, repetition_penalty_apply

from typing import Union, Callable, Optional
import torch.nn.functional as nnf
import torch

def generate_nucleus_sampling(
    model: Union[ClipCapModel, ClipCapModelPrefixOnly],
    tokenizer: Callable,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    top_p: float = 0.8,
    top_k: int = 0,
    temperature: float = 1.0
):

    stop_token = tokenizer.encode(tokenizer.eos_token)[0]
    tokens = None
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_input_embeddings()(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        embeds_init = embeds
        for _ in range(number_to_generate):
            tokens = text_prefix_tokens
            embeds = embeds_init
            for _ in range(entry_length):
                outputs = model.language_model(inputs_embeds=embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                if top_k == 0:
                    top_k = logits.shape[-1]
                if top_p is None:
                    top_p = 1.0
                    
                p, largest_p_idx = nnf.softmax(logits, dim=-1).topk(top_k, dim=-1)
                cumulative_p = p.cumsum(dim=-1)
                threshold_repeated = top_p + torch.zeros((len(p), 1)).to(embeds.device)
                idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k-1).squeeze()
                cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
                censored_p = (cumulative_p <= cutoffs[:, None]) * p
                renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

                final_p = torch.zeros_like(logits)
                row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1,top_k).to(embeds.device)
                final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

                next_token = torch.multinomial(final_p, num_samples=1)
                
                next_token_embed = model.language_model.get_input_embeddings()(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                
                if stop_token == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
        
            generations.append(output_text)
    
    return generations