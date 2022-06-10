from clipcap.model.model import ClipCapModel, ClipCapModelPrefixOnly

from clipcap.inference.no_beam import generate_no_beam

from typing import Union, Callable, Optional
import torch

def generate(
    model: Union[ClipCapModel, ClipCapModelPrefixOnly],
    tokenizer: Callable,
    embeddings: torch.Tensor,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 1.0,
    number_to_generate: int = 5,
    text_prefix: Optional[str] = None,
    stop_token: Optional[str] = None,
):
    batch_size = embeddings.shape[0]
    assert batch_size == 1, "Batch size > 1 support coming soon - for now leave embeddings.shape[0] as 1."

    if text_prefix is not None:
        text_prefix = tokenizer.bos_token + text_prefix
    else:
        text_prefix = tokenizer.bos_token

    text_prefix_tokens = tokenizer.encode(text_prefix, return_tensors="pt").expand(batch_size, -1).to(embeddings.device)
    
    with torch.no_grad():
        token_embeddings = model.language_model.get_input_embeddings()(text_prefix_tokens)
        prefix_projections = model.transformer_mapper(embeddings)
    
    inputs_embeds = torch.cat((prefix_projections, token_embeddings), dim=1)

    captions = generate_no_beam(
        model, tokenizer, inputs_embeds,
        number_to_generate=number_to_generate,
        text_prefix_tokens=text_prefix_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )

    return captions

# mask = tokens.ge(0)
# tokens[~mask] = 0

# prefix_projections = self.transformer_mapper(embeddings)

# mask = torch.cat((torch.ones(prefix_projections.shape[:-1], dtype=torch.bool, device=device), mask), dim=1)  # adding prefix mask
# attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)


# caption_tokens = tokenize(captions).to(embeddings.device)

# with torch.no_grad():
#     text_features = encode_method.model.encode_text(caption_tokens)

#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     media_features /= media_features.norm(dim=-1, keepdim=True)

#     similarities = text_features.cpu().numpy() @ media_features.cpu().numpy().T
#     mean_similarity = float(np.mean(similarities))
#     best_idx = int(np.argmax(similarities))
#     similarities = similarities.tolist()

    
# best = captions[best_idx]
# for caption, similarity in zip(captions, similarities):
#     print("sim", similarity, "caption", caption)
# print("mean sim", mean_similarity)
# print("best", best)