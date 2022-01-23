from transformers import GPT2LMHeadModel
import torch

class GPT2(object):
    def __init__(self, gpt_type: str = "gpt2-xl", **huggingface_kwargs):
        self.model = GPT2LMHeadModel.from_pretrained(gpt_type, **huggingface_kwargs)
    
    def get_embedding_size(self) -> int:
        return self.model.transformer.wte.weight.shape[1]
    
    def get_embedding_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.transformer.wte(tokens)
    
    def __call__(self, inputs_embeds: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)
    
    def eval(self):
        return self.model.eval()
    
    def train(self, mode: bool = True):
        return self.model.train(mode)