"""main module combines distributor, runner, reader, mapper, writer to produce clip embeddings"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import math

from clipcap.preprocess.distributor import PysparkDistributor, SequentialDistributor
from clipcap.preprocess.reader import folder_to_keys, FilesReader, WebdatasetReader
from clipcap.preprocess.writer import NumpyWriter, save_config
from clipcap.preprocess.args import add_preprocess_args
from clipcap.preprocess.mapper import EncoderMapper
from clipcap.preprocess.runner import Runner

from clipcap.encoders.base import get_encoder_from_config
from clipcap.encoders.args import add_encoder_args
from clipcap.encoders.config import EncoderConfig


def preprocess(args: Namespace) -> int:
    """
    Main preprocess code requiring parsed args as an input.
    """

    if args.input_format == "webdataset":
        from braceexpand import braceexpand
        datasets = args.input_dataset.split(",")
        input_dataset = [uri for dataset in datasets for uri in list(braceexpand(dataset))]
    else:
        input_dataset = args.input_dataset

    encoder_config = EncoderConfig.from_args(args)
    encoder_model, sample_processor = get_encoder_from_config(encoder_config, device=args.device)
    save_config(encoder_config, args.output_folder)
    
    if args.output_partition_count is None:
        if args.input_format == "files":
            keys, _, _ = folder_to_keys(
                input_dataset, args.media_file_extensions.lower().split(",")
            )
            sample_count = len(keys)
        elif args.input_format == "webdataset":
            sample_count = len(input_dataset) * args.wds_samples_per_file
        else:
            print("Unsupported input_format")
            return

        if sample_count == 0:
            print("no samples found")
            return
        else:
            print(f"The number of samples has been estimated to be {sample_count}")

        output_partition_count = math.ceil(sample_count / args.write_batch_size)
    else:
        output_partition_count = args.output_partition_count

    def reader_builder(sampler):
        if args.input_format == "files":
            return FilesReader(
                sampler,
                sample_processor,
                input_dataset,
                args.media_file_extensions.lower().split(","),
                args.batch_size,
                args.workers
            )
        elif args.input_format == "webdataset":
            return WebdatasetReader(
                sampler,
                sample_processor,
                input_dataset,
                args.batch_size,
                args.workers,
                wds_media_key=args.wds_media_key,
                wds_caption_key=args.wds_caption_key,
                cache_path=args.wds_cache_path,
            )
        else:
            raise ValueError(f"Unknown input_format: {args.input_format}")

    def mapper_builder():
        return EncoderMapper(
            model=encoder_model,
            device=args.device
        )

    def writer_builder(i):
        return NumpyWriter(
            partition_id=i,
            output_folder=args.output_folder,
            output_partition_count=output_partition_count,
        )

    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        output_partition_count=output_partition_count,
    )

    if args.distribution_strategy == "sequential":
        distributor = SequentialDistributor(runner, output_partition_count)
    elif args.distribution_strategy == "pyspark":
        distributor = PysparkDistributor(runner, output_partition_count)
    distributor()

    return 0


def start_preprocess() -> int:
    """
    Main preprocess function.
    """
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser = add_preprocess_args(parser)
    parser = add_encoder_args(parser)
    args = parser.parse_args()
    return preprocess(args)


if __name__ == "__main__":
    exit(start_preprocess())