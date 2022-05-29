from argparse import ArgumentParser
import pytorch_lightning as pl

from clipcap.train.dataloader import get_dataloader
from clipcap.train.callback import CheckpointSaver
from clipcap.train.args import add_training_args

from clipcap.model import (
    ClipCapModel, ClipCapModelPrefixOnly,
    add_model_args, model_config_from_args,
    training_config_from_args
)


def train(args: ArgumentParser) -> int:
    """ Starts the main training process. """

    # print(f'Using pytorch version {torch.__version__}')
    # print('Args: ', locals())

    # Prepare training datasets.
    dataloader = get_dataloader(
        data_path=args.input_dataset,
        language_model=args.language_model,
    )

    total_steps = len(dataloader) * args.epochs

    model_config = model_config_from_args(args)
    training_config = training_config_from_args(args)
    
    if not args.train_language_model:
        model = ClipCapModelPrefixOnly(model_config)
        # print("Train only Prefix.")
    else:
        model = ClipCapModel(model_config)
        # print("Train both Prefix and Language Model.")
    
    model.set_training_config(training_config)

    # Easier to use GPU args. `-1` = use all, `0` = use gpu 0, `0,1` = use gpus 1 and 2 etc.
    if "," not in args.device:
        args.device = int(args.device)
        if args.device != -1:
            args.device = [args.device]
    
    # Create `CheckpointSaver` as a trainer callback instance.
    checkpoint_saver = CheckpointSaver(
        args.output_folder,
        args.checkpoint_filename_prefix,
        save_every_n_epochs=args.checkpoint_save_frequency
    )

    # Save model config for future loading / reference.
    checkpoint_saver.save_config(model_config, training=False)
    checkpoint_saver.save_config(training_config, training=True)

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
    Main preprocess function, using environment args.
    """
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    exit(start_training())