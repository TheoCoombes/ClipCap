from clipcap.model.model import ClipCapModel, ClipCapModelPrefixOnly

from clipcap.inference.base import *

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
    text_prefix: Optional[str] = None
):
    batch_size = embeddings.shape[0]
    assert batch_size == 1, "Batch size > 1 support coming soon - for now leave embeddings.shape[0] as 1."

    if text_prefix is not None:
        text_prefix_tokens = tokenizer.encode(text_prefix, return_tensors="pt").expand(batch_size, -1).to(embeddings.device)
    else:
        text_prefix_tokens = None
    
    with torch.no_grad():
        prefixes = model.transformer_mapper(embeddings)

    captions = generate_no_beam(
        model, tokenizer, prefixes,
        number_to_generate=number_to_generate,
        text_prefix_tokens=text_prefix_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )

    return captions


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