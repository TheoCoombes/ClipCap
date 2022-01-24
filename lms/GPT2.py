from transformers import GPT2LMHeadModel
import torch

class GPT2(GPT2LMHeadModel):
    @classmethod
    def create(cls, model_variant: str = "gpt2-xl", **huggingface_kwargs):
        return cls.from_pretrained(model_variant, **huggingface_kwargs)
    
    def get_embedding_size(self) -> int:
        return self.transformer.wte.weight.shape[1]
    
    def get_embedding_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(tokens)
    
    def call(self, inputs_embeds: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)