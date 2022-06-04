from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.nn import functional as F
from typing import Optional, Tuple, Callable, List
import pytorch_lightning as pl
import torch

from clipcap.model.mapper import TransformerMapper, TransformerMapperWindowed
from clipcap.model.config import Config, TrainingConfig

def get_tokenizer(language_model_name: str, **huggingface_kwargs):
    return AutoTokenizer.from_pretrained(language_model_name, **huggingface_kwargs)

class ClipCapModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config

        self.language_model = AutoModelForCausalLM.from_pretrained(self.config.language_model)
        self.lm_embedding_size = self.language_model.get_input_embeddings().weight.shape[1]

        if self.config.encoder_config.use_windowed_embeddings:
            self.transformer_mapper = TransformerMapperWindowed(
                encoder_embedding_size=self.config.encoder_config.encoder_embedding_size,
                lm_embedding_size=self.lm_embedding_size,
                prefix_length=self.config.prefix_length,
                projection_length=self.config.projection_length,
                window_size=(self.config.encoder_config.window_size + 1),
                use_pos_embeddings=self.config.use_positional_embeddings,
                num_heads=self.config.transformer_attention_heads,
                num_layers=self.config.transformer_layers
            )
        else:
            self.transformer_mapper = TransformerMapper(
                encoder_embedding_size=self.config.encoder_config.encoder_embedding_size,
                lm_embedding_size=self.lm_embedding_size,
                prefix_length=self.config.prefix_length,
                projection_length=self.config.projection_length,
                num_heads=self.config.transformer_attention_heads,
                num_layers=self.config.transformer_layers
            )

    def forward(self, tokens: torch.Tensor, embeddings: torch.Tensor, mask: torch.Tensor):
        # Get text and prefix embeddings.
        token_embeddings = self.language_model.get_input_embeddings()(tokens)
        prefix_projections = self.transformer_mapper(embeddings)

        # Concatenate to form LM input.
        inputs_embeds = torch.cat((prefix_projections, token_embeddings), dim=1)

        # Create and concatenate prefix mask to caption mask.
        prefix_mask = torch.ones(prefix_projections.shape[:-1], dtype=torch.bool, device=mask.device)
        mask = torch.cat((prefix_mask, mask), dim=1)
        
        # Call language model.
        out = self.language_model(inputs_embeds=inputs_embeds, attention_mask=mask)

        return out
    
    def set_training_config(self, training_config: TrainingConfig, reinit_optims: bool = False) -> None:
        """ Stores a dict containing deepspeed args and loss rates etc. (see config.py) """
        self.config.training_config = training_config
        self.save_hyperparameters(self.config.to_dict())
        if reinit_optims:
            self.configure_optimizers()
    
    def configure_optimizers(self) -> dict:
        """ Returns a dict containing the model's optimizer and loss rate scheduler. """

        assert self.config.training_config is not None, "You must first use `set_training_config` before training."

        if self.config.training_config.use_deepspeed_optimisers:
            from deepspeed.ops.adam import FusedAdam
            optimizer = FusedAdam(self.parameters(), lr=self.config.training_config.optimizer_lr, adam_w_mode=True)
        else: 
            from torch.optim import AdamW
            optimizer = AdamW(self.parameters(), lr=self.config.training_config.optimizer_lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.training_config.scheduler_warmup_steps,
            num_training_steps=self.config.training_config.total_steps
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """
        The model's main training step.
        `batch` contains a tuple of the caption's tokens and the encoder's embedding (prefix). [see the `preprocess` folder]
        """

        tokens, embeds = batch

        # Create mask (with -1 pads)
        mask = tokens.ge(0)
        tokens[~mask] = 0

        outputs = self(tokens, embeds, mask)

        logits = outputs.logits[:, self.config.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        self.log("loss", loss.float())
        
        return loss


class ClipCapModelPrefixOnly(ClipCapModel):
    def parameters(self):
        return self.transformer_mapper.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self