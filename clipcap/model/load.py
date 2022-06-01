from clipcap.model.model import ClipCapModel, ClipCapModelPrefixOnly, get_tokenizer
from clipcap.encoders.config import EncoderConfig
from clipcap.model.config import Config

from typing import Union, Tuple, Callable
import torch
import yaml

def load(model_path: str, config_path: str, device: str = "cpu",
         from_checkpoint: bool = False) -> Tuple[Union[ClipCapModel, ClipCapModelPrefixOnly], Callable]:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    print(raw_config)
    
    # Remove old training config data from past training runs.
    if from_checkpoint and raw_config["training_config"] is not None:
        raw_config["training_config"] = None
    
    raw_config["encoder_config"] = EncoderConfig(**raw_config["encoder_config"])
    config = Config(**raw_config)
    
    if config.train_language_model:
        model_cls = ClipCapModel
    else:
        model_cls = ClipCapModelPrefixOnly
    
    model = model_cls(config)

    # Load state dict.
    state_dict = torch.load(model_path, map_location="cpu")

    if from_checkpoint:
        state_dict = state_dict["state_dict"]
    
    model.load_state_dict(state_dict)
    
    # Set to eval and load onto device.
    model = model.eval()
    model = model.to(device)

    # Fetch tokenizer.
    tokenizer = get_tokenizer(config.language_model)

    return model, tokenizer