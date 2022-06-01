from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import pytorch_lightning as pl
from pathlib import Path
import yaml

from clipcap.train.dataloader import get_dataloader
from clipcap.train.callback import CheckpointSaver
from clipcap.train.args import add_training_args
from clipcap.encoders import EncoderConfig

from clipcap.model import (
    ClipCapModel, ClipCapModelPrefixOnly,
    add_model_args, Config, TrainingConfig
)


def train(args: Namespace) -> int:
    """ Starts the main training process. """

    # print(f'Using pytorch version {torch.__version__}')
    # print('Args: ', locals())

    if args.deepspeed:
        assert args.deepspeed_strategy is not None, "--deepspeed-strategy must not be None if --deepspeed is enabled."

    with open(Path(args.input_dataset) / "encoder_config.yaml", "r") as f:
        encoder_config_raw = yaml.safe_load(f)
    
    encoder_config = EncoderConfig(**encoder_config_raw)

    # Prepare training datasets.
    dataloader, encoder_embedding_size = get_dataloader(
        data_path=args.input_dataset,
        language_model=args.language_model,
        batch_size=args.batch_size
    )

    # Add more args to namespace. [TODO better way to do this?]
    encoder_config.encoder_embedding_size = encoder_embedding_size
    args.total_steps = len(dataloader) * args.epochs

    model_config = Config.from_args(args)
    model_config.training_config = TrainingConfig.from_args(args)
    model_config.encoder_config = encoder_config
    
    if not args.train_language_model:
        model = ClipCapModelPrefixOnly(model_config)
        # print("Train only Prefix.")
    else:
        model = ClipCapModel(model_config)
        # print("Train both Prefix and Language Model.")

    # Easier to use GPU args. `-1` = use all, `0` = use gpu 0, `0,1` = use gpus 1 and 2 etc.
    if "," not in args.device:
        args.device = int(args.device)
        if args.device != -1:
            args.device = [args.device]
    
    # Create `CheckpointSaver` as a trainer callback instance.
    checkpoint_saver = CheckpointSaver(
        args.output_folder,
        args.checkpoint_filename_prefix,
        save_every_n_epochs=args.checkpoint_save_frequency,
        use_deepspeed=args.enable_deepspeed
    )

    # Save model config for future loading / reference.
    checkpoint_saver.save_config(model_config.to_dict())

    if args.enable_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project)
    else:
        logger = None

    # Create trainer class.
    trainer = pl.Trainer(
        gpus=args.device,
        max_epochs=args.epochs,
        callbacks=[checkpoint_saver],
        strategy=args.deepspeed_strategy,
        precision=args.fp_precision,
        logger=logger,
        log_every_n_steps=args.logging_frequency
    )

    # Run training process.
    trainer.fit(model, dataloader)

    # Save final checkpoint.
    checkpoint_saver.save_final_checkpoint(trainer)

    return 0


def start_training() -> int:
    """
    Main training function.
    """
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser = add_training_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    exit(start_training())