from clipcap.model import ClipCapModel, ClipCapModelPrefixOnly

from typing import Union, Callable, Optional
import torch.nn.functional as nnf
import random
import torch

# From https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def repetition_penalty_apply(logits, tokens, penalty):
    tok_logits = torch.gather(logits, -1, tokens)
    tok_logits = torch.where(tok_logits < 0, tok_logits * penalty, tok_logits / penalty)
    logits.scatter_(-1, tokens, tok_logits)
    return logits

def sentence_length_penalty_apply(logits: torch.Tensor, tokens: torch.Tensor, stop_token: int,
                                  current_length: int, desired_length: int, length_factor: float) -> torch.Tensor:
    penalty = (current_length / desired_length) * length_factor

    tok_logits = torch.gather(logits, -1, tokens)
    tok_logits = torch.where(tok_logits == stop_token, tok_logits * penalty, tok_logits)

    logits.scatter_(-1, tokens, tok_logits)
    
    return logits

def generate_beam(
    model: Union[ClipCapModel, ClipCapModelPrefixOnly],
    tokenizer: Callable,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    beam_size: int = 5,
    entry_length: int = 67,
    temperature: float = 1.0
):

    stop_token = tokenizer.encode(tokenizer.eos_token)[0]
    tokens = None
    scores = None

    seq_lengths = torch.ones(beam_size, device=embeds.device)
    has_stopped = torch.zeros(beam_size, dtype=torch.bool, device=embeds.device)
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_input_embeddings()(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        for _ in range(number_to_generate):
            for _ in range(entry_length):
                outputs = model.language_model(inputs_embeds=embeds)
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
                    logits[has_stopped] = -float('inf')
                    logits[has_stopped, 0] = 0

                    scores_sum = scores[:, None] + logits
                    seq_lengths[~has_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)

                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='trunc')
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)

                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)

                    embeds = embeds[next_tokens_source]
                    scores = scores_sum_average * seq_lengths

                    has_stopped = has_stopped[next_tokens_source]
                
                next_token_embed = model.language_model.get_input_embeddings()(next_tokens.squeeze()).view(embeds.shape[0], 1, -1)
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                has_stopped = has_stopped + next_tokens.eq(stop_token).squeeze()
                if has_stopped.all():
                    break
                
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]

            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order][0]

            generations.append(output_texts)
    
    return generations


def generate_nucleus_sampling(
    model: Union[ClipCapModel, ClipCapModelPrefixOnly],
    tokenizer: Callable,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    top_p: float = 0.8,
    top_k = None,
    temperature: float = 1.0
):

    stop_token = tokenizer.encode(tokenizer.eos_token)[0]
    tokens = None
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_input_embeddings()(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        for i in range(number_to_generate):
            for _ in range(entry_length):
                outputs = model.language_model(inputs_embeds=embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                if top_k is None:
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


def generate_no_beam(
    model: Union[ClipCapModel, ClipCapModelPrefixOnly],
    tokenizer: Callable,
    embeds: torch.Tensor,
    number_to_generate: int = 5,
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
    print(f'Generate_no_beam (repetition_penalty: {repetition_penalty:.2f})')
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_input_embeddings()(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        embeds_init = embeds
        for top_p in [0.7, 0.8, 0.9] * 3:
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
                if repetition_penalty != 1.0 and tokens is not None:
                    tokens1 = tokens[0, :] # assuming batch size of 1
                    logits = repetition_penalty_apply(logits, tokens1, repetition_penalty)

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
        
            generations.append(output_text)
    
    return generations

