from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, List
import torch


class T0(T5ForConditionalGeneration):
    @classmethod
    def create(cls, model_variant: str = "bigscience/T0pp", **huggingface_kwargs):
        return cls.from_pretrained(model_variant, **huggingface_kwargs)

    def get_embedding_size(self) -> int:
        return self.encoder.embed_tokens.weight.shape[1]
    
    def get_embedding_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_tokens(tokens)
    
    def call(self, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self(inputs_embeds=inputs_embeds, decoder_inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)


class T0_Tokenizer(T5Tokenizer):
    @classmethod
    def create(cls, model_variant: str = "bigscience/T0pp", **huggingface_kwargs):
        return cls.from_pretrained(model_variant, **huggingface_kwargs)
    
    def encode_text(self, text: str, max_token_length: Optional[int] = None) -> List[int]:
        tokens = self.encode(text)
        if max_token_length is not None:
            tokens = tokens[:max_token_length]
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> str:
        return self.decode(tokens)