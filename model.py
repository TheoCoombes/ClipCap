from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as nnf
from typing import Optional, Tuple
from functools import lru_cache
import pytorch_lightning as pl
import torch.nn as nn
import torch

from layers import TransformerMapper, MLP
from lms import GPT2, GPTJ, T0


class CLIPCaptionModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Save hparams (see `train.py` for arguments).
        self.save_hyperparameters(kwargs)
        
        if not self.hparams.use_deepspeed:
            # Deepspeed will load the models using configure_sharded_model.
            self.init_models()

    def init_models(self):
        """ Loads the language model and the mapping layers. """

        # load language model.
        if self.hparams.language_model_type == "gpt2":
            self.language_model = GPT2.create(self.hparams.language_model_variant)
        elif self.hparams.language_model_type in ("gptj", "gpt-j"):
            self.language_model = GPTJ.create(self.hparams.language_model_variant)
        elif self.hparams.language_model_type in ("t0", "t5"):
            self.language_model = T0.create(self.hparams.language_model_variant)
        else:
            raise ValueError(f"invalid language model type '{self.hparams.language_model_type}' (expected 'gpt-j' / 'gpt2' / 't0' / 't5')")

        # Get the size of the LM's embeddings.
        self.lm_embedding_size = self.language_model.get_embedding_size()

        # load mapping layers
        if self.hparams.mapping_type == 'mlp':
            self.clip_project = MLP((
                self.hparams.prefix_size,
                (self.lm_embedding_size * self.hparams.prefix_length) // 2,
                self.lm_embedding_size * self.hparams.prefix_length
            ))
        elif self.hparams.mapping_type == 'transformer':
            self.clip_project = TransformerMapper(
                dim_clip=self.hparams.prefix_size,
                dim_embedding=self.lm_embedding_size,
                prefix_length=self.hparams.prefix_length,
                clip_length=self.hparams.clip_prefix_length,
                num_heads=self.hparams.num_attention_heads,
                num_layers=self.hparams.num_layers
            )
        else:
            raise ValueError(f"invalid mapping type: '{self.hparams.mapping_type}' (choose from 'mlp'/'transformer')")


    def configure_sharded_model(self):
        """ [deepspeed] Shards the models on initialization to prevent OOM errors. """
        return self.init_models()


    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.language_model.get_embedding_text(tokens)

        prefix_projections = self.clip_project(prefix).view(-1, self.hparams.prefix_length, self.lm_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = torch.zeros(tokens.shape[0], self.prefix_length, dtype=torch.int64)
            labels = torch.cat((dummy_token, tokens), dim=1)
        
        out = self.language_model.call(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)

        return out
    
    def configure_optimizers(self):
        if self.use_deepspeed:
            from deepspeed.ops.adam import FusedAdam
            optimizer = FusedAdam(self.parameters(), lr=self.hparams.optimizer_lr, adam_w_mode=True)
        else:  
            optimizer = AdamW(self.parameters(), lr=self.hparams.optimizer_lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.scheduler_warmup_steps,
            num_training_steps=self.hparams.total_steps
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], _):
        tokens, mask, prefix = batch

        # Fix for custom dataloader.
        tokens = tokens[0]
        mask = mask[0]
        prefix = prefix[0]

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
    
    def init_models(self):
        super().init_models()

        self.language_model = self.language_model.eval()

        for param in self.language_model.parameters():
            param.requires_grad = False