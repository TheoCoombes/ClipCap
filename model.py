from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as nnf
from typing import Optional, Tuple
import pytorch_lightning as pl
import torch.nn as nn
import torch

from layers import TransformerMapper, MLP
from lms import GPT2


class CLIPCaptionModel(pl.LightningModule):
    def __init__(self, language_model: GPT2, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
            num_layers: int = 8, mapping_type: str = 'mlp', optimizer_lr: float = 2e-5, num_warmup_steps: int = 5000):
        
        super().__init__()

        self.optimizer_lr = optimizer_lr
        self.num_warmup_steps = num_warmup_steps
        self.prefix_length = prefix_length

        self.language_model = language_model
        self.lm_embedding_size = self.language_model.get_embedding_size()

        if mapping_type == 'mlp':
            self.clip_project = MLP(
                (prefix_size, (self.lm_embedding_size * prefix_length) // 2, self.lm_embedding_size * prefix_length)
            )
        elif mapping_type == 'transformer':
            self.clip_project = TransformerMapper(
                prefix_size, self.lm_embedding_size, prefix_length, clip_length, num_layers
            )
        else:
            raise ValueError(f"invalid mapping type: '{mapping_type}' (choose from 'mlp'/'transformer')")

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
        
        out = self.language_model.call(input_embeds=embedding_cat, labels=labels, attention_mask=mask)

        return out
    
    def setup(self, stage=None):
        dataloader = self.train_dataloader()
        
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)

        self.total_steps = (len(dataloader.dataset) // tb_size) // ab_size
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.optimizer_lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.total_steps
        )
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        tokens, mask, prefix = batch

        outputs = self(tokens, prefix, mask)

        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        self.log("loss", loss.item())

        return loss


class CLIPCaptionPrefix(CLIPCaptionModel):
    def parameters(self):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self