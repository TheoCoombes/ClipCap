import tokenizers
from torch.utils.data import IterableDataset, DataLoader
from embedding_reader import EmbeddingReader
from typing import Tuple, List
import torch
import math

from clipcap.model import get_tokenizer


class EmbedDataset(IterableDataset):
    """
    Streams preprocessed embeddings with the EmbeddingReader class from https://github.com/rom1504/embedding-reader.
    """

    def __init__(self, data_path: str = "./dataset/", language_model: str = "gpt2-xl", batch_size: int = 256,
                 reader_max_piece_size: int = 50, reader_parallel_pieces: int = 10, max_token_length: int = 64) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer(language_model)

        self.batch_size = batch_size
        self.reader_max_piece_size = reader_max_piece_size
        self.reader_parallel_pieces = reader_parallel_pieces
        self.max_token_length = max_token_length

        if not data_path.endswith("/"):
            data_path += "/" # Keep string to allow s3 etc.
        
        embedding_folder = data_path + "embeddings"
        captions_folder = data_path + "captions"

        self.reader = EmbeddingReader(
            embeddings_folder=embedding_folder,
            metadata_folder=captions_folder,
            file_format="parquet_npy",
            meta_columns=['caption'],
        )

        self.encoder_embedding_size = self.reader.dimension #[-1]
    
    def pad_tokens(self, tokens: List[int]):
        tokens = torch.tensor(tokens)
        padding = self.max_token_length - tokens.shape[0]

        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1), dim=0)
        elif padding < 0:
            tokens = tokens[:self.max_token_length]
        
        return tokens
    
    def __iter__(self):
        for batch, metadata in self.reader(
            batch_size=self.batch_size, start=0, end=self.reader.count, max_piece_size=self.reader_max_piece_size,
            parallel_pieces=self.reader_parallel_pieces, show_progress=False
        ):
            batch = torch.from_numpy(batch)
            captions = metadata["caption"].to_list()

            tokens = self.tokenizer.batch_encode_plus(captions)["input_ids"]
            tokens = torch.cat([self.pad_tokens(sample).unsqueeze(0) for sample in tokens], dim=0)

            yield tokens, batch

    def __len__(self) -> int:
        return math.ceil(self.reader.count / self.batch_size)


def get_dataloader(
    data_path: str = "./dataset/",
    language_model: str = "gpt2-xl",
    batch_size: int = 256
) -> Tuple[DataLoader, int]:
    """
    Initializes an EmbedDataset class and returns the appropriate torch DataLoader along with the encoder's embedding size for training reference.
    """
    dataset = EmbedDataset(
        data_path=data_path,
        language_model=language_model,
        batch_size=batch_size
    )

    # Batching is already implemented in the EmbedDataset.
    collate_fn = lambda x: x[0]

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    return dataloader, dataset.encoder_embedding_size