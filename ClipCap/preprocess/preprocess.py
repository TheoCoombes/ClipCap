"""main module combines distributor, runner, reader, mapper, writer to produce clip embeddings"""

from braceexpand import braceexpand
from argparse import ArgumentParser

from ClipCap.preprocess.distributor import PysparkDistributor, SequentialDistributor
from ClipCap.preprocess.reader import folder_to_keys, FilesReader, WebdatasetReader
from ClipCap.preprocess.args import add_preprocess_args
from ClipCap.preprocess.mapper import EncoderMapper
from ClipCap.preprocess.writer import NumpyWriter
from ClipCap.preprocess.runner import Runner

from ClipCap.encoders import get_encoder_from_args, add_encoder_args


def preprocess(args: ArgumentParser) -> int:
    """
    Main preprocess code requiring parsed args as an input.
    """

    if args.input_format == "webdataset":
        input_dataset = list(braceexpand(args.input_dataset))
    else:
        input_dataset = args.input_dataset

    encoder_model, sample_processor = get_encoder_from_args(args)
    
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
            print("no sample found")
            return
        else:
            print(f"The number of samples has been estimated to be {sample_count}")

        args.output_partition_count = int(sample_count / args.write_batch_size) + 1

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
            normalize=args.normalize_embeddings,
            device=args.device
        )

    def writer_builder(i):
        return NumpyWriter(
            partition_id=i,
            output_folder=args.output_folder,
            output_partition_count=args.output_partition_count,
        )

    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        output_partition_count=args.output_partition_count,
    )

    if args.distribution_strategy == "sequential":
        distributor = SequentialDistributor(runner, args.output_partition_count)
    elif args.distribution_strategy == "pyspark":
        distributor = PysparkDistributor(runner, args.output_partition_count)
    distributor()

    return 0


def start_preprocess() -> int:
    """
    Main preprocess function, using environment args.
    """
    args = ArgumentParser()
    args = add_preprocess_args(args)
    args = add_encoder_args(args)
    return preprocess(args)


if __name__ == "__main__":
    exit(start_preprocess())