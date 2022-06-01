from clipcap.model.model import ClipCapModel, ClipCapModelPrefixOnly, get_tokenizer
from clipcap.encoders import EncoderConfig
from clipcap.model.config import Config

from typing import Union, Tuple, Callable
import torch
import yaml

def load(model_path: str, config_path: str, device: str = "cpu",
         from_checkpoint: bool = False) -> Tuple[Union[ClipCapModel, ClipCapModelPrefixOnly], Callable]:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # Remove old training config data from past training runs.
    if from_checkpoint and raw_config["training_config"] is not None:
        raw_config["training_config"] = None
    
    raw_config["encoder_config"] = EncoderConfig(**raw_config["encoder_config"])
    config = Config(**raw_config)
    
    if config.train_language_model:
        model_cls = ClipCapModel
    else:
        model_cls = ClipCapModelPrefixOnly
    
    if from_checkpoint:
        model = model_cls.load_from_checkpoint(model_path)
    else:
        # Is a state_dict.
        model = model_cls(config.to_dict())
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    model = model.eval()
    model = model.to(device)
    tokenizer = get_tokenizer(config.language_model)

    return model, tokenizer