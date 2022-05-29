"""writer module saves embeddings"""

from io import BytesIO
import fsspec
import math


class OutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""

    def __init__(self, output_folder, partition_id, output_partition_count):
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.embed_folder = output_folder + "/embeddings"
        self.captions_folder = output_folder + "/captions"
        self.batch_num = partition_id
        self.oom_partition_count = int(math.log10(output_partition_count)) + 1

        self.fs.makedirs(self.embed_folder, exist_ok=True)
        self.fs.makedirs(self.captions_folder, exist_ok=True)

        self.batch_count = 0
        self.__init_batch()

    def __init_batch(self):
        self.embeddings = []
        self.captions = []
        self.batch_count = 0

    def add(self, sample):
        """
        add to buffers the image embeddings, text embeddings, and meta
        """

        self.batch_count += sample["image_embs"].shape[0]
        self.embeddings.append(sample["embeddings"])
        self.captions.extend(sample["text"])

    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pandas as pd  # pylint: disable=import-outside-toplevel

        data_lists = []
        data_columns = []
        batch_num_str = str(self.batch_num).zfill(self.oom_partition_count)

        embedding_mat = np.concatenate(self.image_embeddings)
        output_path_embeds = self.embed_folder + "/embeds_" + batch_num_str

        with self.fs.open(output_path_embeds + ".npy", "wb") as f:
            npb = BytesIO()
            np.save(npb, embedding_mat)
            f.write(npb.getbuffer())

        data_lists.append(self.captions)
        data_columns.append("caption")

        df = pd.DataFrame(data=list(zip(*data_lists)), columns=data_columns)

        output_path_metadata = self.captions_folder + "/captions_" + batch_num_str + ".parquet"
        with self.fs.open(output_path_metadata, "wb") as f:
            df.to_parquet(f)

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        self.__init_batch()


class NumpyWriter:
    """the numpy writer writes embeddings to folders img_emb, text_emb, and metadata"""

    def __init__(self, partition_id, output_folder, output_partition_count):
        self.sink = OutputSink(
            output_folder, partition_id, output_partition_count
        )

    def __call__(self, batch):
        self.sink.add(batch)

    def flush(self):
        self.sink.flush()
