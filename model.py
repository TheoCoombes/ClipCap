from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Optional, Tuple, Union
from torch.nn import functional as nnf
import pytorch_lightning as pl
import torch.nn as nn
import torch

from layers import TransformerMapper, MLP
from lms import GPT2, GPTJ, T0


class CLIPCaptionModel(pl.LightningModule):
    def __init__(self, language_model: Union[GPT2, GPTJ, T0], prefix_length: int = 40, clip_prefix_length: int = 40,
                 prefix_size: int = 512, num_layers: int = 8, mapping_type: str = 'mlp', optimizer_lr: float = 2e-5,
                 num_warmup_steps: int = 5000, total_steps=None, use_8_bit_optimizers: bool = False):
        
        super().__init__()

        # Save hparams for easy model loading.
        self.save_hyperparameters(ignore=["language_model"])
        
        # Save hparams as class attributes for better readability.
        self.language_model = language_model
        self.prefix_length = prefix_length
        self.clip_prefix_length = clip_prefix_length
        self.prefix_size = prefix_size
        self.num_layers = num_layers
        self.mapping_type = mapping_type
        self.optimizer_lr = optimizer_lr
        self.num_warmup_steps = num_warmup_steps
        self.total_steps = total_steps # TODO - find a better workaround finding the total step amount (for `get_linear_schedule_with_warmup`)
        self.use_8_bit_optimizers = use_8_bit_optimizers
        
        self.lm_embedding_size = self.language_model.get_embedding_size()

        if self.mapping_type == 'mlp':
            self.clip_project = MLP(
                (self.prefix_size, (self.lm_embedding_size * self.prefix_length) // 2, self.lm_embedding_size * self.prefix_length)
            )
        elif self.mapping_type == 'transformer':
            self.clip_project = TransformerMapper(
                self.prefix_size, self.lm_embedding_size, self.prefix_length, self.clip_prefix_length, self.num_layers
            )
        else:
            raise ValueError(f"invalid mapping type: '{self.mapping_type}' (choose from 'mlp'/'transformer')")

    def get_dummy_token(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.language_model.get_embedding_text(tokens)

        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.lm_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0])
            labels = torch.cat((dummy_token, tokens), dim=1)
        
        out = self.language_model.call(embedding_cat, labels, mask)

        return out
    
    def configure_optimizers(self):
        if self.use_8_bit_optimizers:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(self.parameters(), lr=self.optimizer_lr)
        else:  
            optimizer = AdamW(self.parameters(), lr=self.optimizer_lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.total_steps
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        tokens, mask, prefix = batch
        outputs = self(tokens, prefix, mask)

        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        
        return loss


class CLIPCaptionPrefixOnly(CLIPCaptionModel):
    def parameters(self):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self
