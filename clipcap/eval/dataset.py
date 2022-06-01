from typing import Callable
from pathlib import Path

class EvalDataset(object):
    def __init__(
        self,
        sample_processor: Callable,
        folder: str,
    ):
        super().__init__()

        folder = Path(folder)
        self.media_files = list(folder.glob("*"))
        self.sample_processor = sample_processor

    def __len__(self):
        return len(self.media_files)

    def __iter__(self):
        for media_file in self.media_files:
            data_tensor = self.sample_processor(media_file)
            yield media_file.name, data_tensor