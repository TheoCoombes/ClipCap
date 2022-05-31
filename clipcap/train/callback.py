from pytorch_lightning import Callback, Trainer
from typing import Optional
from pathlib import Path
import yaml

class CheckpointSaver(Callback):
    def __init__(self, output_path: str = "./checkpoints/", filename_prefix: str = "clipclap_demo",
                 save_every_n_epochs: int = 1, use_deepspeed: bool = False) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.output_path = output_path
        self.filename_prefix = filename_prefix
        self.save_every_n_epochs = save_every_n_epochs
        self.use_deepspeed = use_deepspeed
    
    def save_config(self, config: dict) -> None:
        with open(self.output_path / f"{self.filename_prefix}_config.yaml", 'w+') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def on_epoch_end(self, trainer: Trainer, _) -> None:
        epoch = trainer.current_epoch
        if epoch % self.save_every_n_epochs == 0:
            output_path = self.output_path / f"{self.filename_prefix}_epoch_{epoch}{'.ckpt' if not self.use_deepspeed else ''}"
            trainer.save_checkpoint(output_path)
    
    def save_final_checkpoint(self, trainer: Trainer) -> None:
        output_path = self.output_path / f"{self.filename_prefix}_final{'.ckpt' if not self.use_deepspeed else ''}"
        trainer.save_checkpoint(output_path)