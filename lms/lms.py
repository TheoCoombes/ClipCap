from transformers import AutoModelForCausalLM
from typing import Optional
import torch

class LanguageModel(AutoModelForCausalLM):
    @classmethod
    def create(cls, model_variant: str = "gpt2-xl", **huggingface_kwargs):
        return cls.from_pretrained(model_variant, **huggingface_kwargs)

    def get_embedding_size(self) -> int:
        return self.transformer.wte.weight.shape[1]
    
    def get_embedding_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(tokens)
    
    def call(self, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)