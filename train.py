from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from pathlib import Path
import torch
import fire

from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from dataset import TokenPrefixDataset
from lms import GPT2, GPTJ, T0

class CheckpointSaver(pl.Callback):
    def __init__(self, output_path: Path, filename_prefix: str, save_every_n_epochs: int = 1,
            save_every_n_steps: Optional[int] = 1000):
        output_path.mkdir(exist_ok=True)

        self.output_path = output_path
        self.filename_prefix = filename_prefix
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps

    def on_epoch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch
        if epoch % self.save_every_n_epochs == 0:
            output_path = self.output_path / f"{self.filename_prefix}_epoch_{epoch}.ckpt"
            trainer.save_checkpoint(output_path)
    
    def on_batch_end(self, trainer: pl.Trainer, _):
        if self.save_every_n_steps is not None:
            current_step = trainer.global_step
            if (current_step % self.save_every_n_steps == 0):
                output_path = self.output_path / f"{self.filename_prefix}_latest.ckpt"
                trainer.save_checkpoint(output_path)


def train(
    data_dir: str = "./train/",
    output_dir: str = "./models/",
    output_filename_prefix: str = "demo_model",
    epochs: int = 10,
    save_every_epochs: int = 1,
    save_every_steps: int = 10000,
    prefix_length: int = 10,
    prefix_size: int = 512,
    clip_prefix_length: int = 10,
    language_model_type = "gpt2",
    language_model_variant = "gpt2-xl",
    batch_size: int = 256,
    only_prefix: bool = False,
    mapping_type: str = "mlp",
    num_layers: int = 8,
    normalize_prefix: bool = False,
    use_8_bit_optimizers: bool = False,
    gpu_devices: str = "0",
    **huggingface_kwargs
):
    dataset = TokenPrefixDataset(data_dir, batch_size=batch_size, normalize_prefix=normalize_prefix)

    total_steps = (len(dataset) // batch_size) * epochs

    if language_model_type == "gpt2":
        language_model = GPT2.create(language_model_variant, **huggingface_kwargs)
    elif language_model_type in ("gptj", "gpt-j"):
        language_model = GPTJ.create(language_model_variant, **huggingface_kwargs)
    elif language_model_type in ("t0", "t5"):
        language_model = T0.create(language_model_variant, **huggingface_kwargs)
    else:
        raise ValueError(f"invalid language model type '{language_model_type}' (expected 'gpt-j' / 'gpt2' / 't0' / 't5')")

    if mapping_type not in ("mlp", "transformer"):
        raise ValueError(f"invalid mapping type '{mapping_type}' (expected 'mlp' or 'transformer')")

    if only_prefix:
        model = CLIPCaptionPrefixOnly(
            language_model, prefix_length=prefix_length, clip_prefix_length=clip_prefix_length,
            prefix_size=prefix_size, num_layers=num_layers, mapping_type=mapping_type,
            total_steps=total_steps, use_8_bit_optimizers=use_8_bit_optimizers
        )
        print("Train only Prefix")
    else:
        model = CLIPCaptionModel(
            language_model, prefix_length=prefix_length, clip_prefix_length=clip_prefix_length, 
            prefix_size=prefix_size, num_layers=num_layers, mapping_type=mapping_type,
            total_steps=total_steps, use_8_bit_optimizers=use_8_bit_optimizers
        )
        print("Train both prefix and language model")

    # Easier to use GPU args. `-1` = use all, `0` = use gpu 0, `0,1` = use gpus 1 and 2 etc.
    if isinstance(gpu_devices, int) and gpu_devices != -1:
        gpu_devices = [gpu_devices]
    
    output_path = Path(output_dir)
    checkpoint_saver = CheckpointSaver(output_path, output_filename_prefix,
        save_every_n_epochs=save_every_epochs, save_every_n_steps=save_every_steps
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # batch_size=1 as the dataset implements batching.

    if "," in str(gpu_devices) or str(gpu_devices) == "-1":
        from pytorch_lightning.plugins import DeepSpeedPlugin
        kwargs = {
            "strategy": DeepSpeedPlugin(stage=3, cpu_offload=True, partition_activations=True)
        }
    else:
        kwargs = {}

    trainer = pl.Trainer(gpus=gpu_devices, max_epochs=epochs, callbacks=[checkpoint_saver], **kwargs)
    trainer.fit(model, dataloader)

    trainer.save_checkpoint(output_path / f"{output_filename_prefix}_final.ckpt")


if __name__ == '__main__':
    fire.Fire(train)
