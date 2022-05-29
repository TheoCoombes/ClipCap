"""Reader module provides files and webdataset readers"""

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from pathlib import Path
import io


def folder_to_keys(folder, media_file_extensions: list):
    """returns a list of keys from a folder of images and text"""
    path = Path(folder)

    text_files = [*path.glob("**/*.txt")]
    text_files = {text_file.stem: text_file for text_file in text_files}

    image_files = [list(path.glob(f"**/*.{filetype}")) for filetype in media_file_extensions]
    image_files = [file for filetype in image_files for file in filetype] # flatten (overcomplicated?)
    image_files = {image_file.stem: image_file for image_file in image_files}

    keys = None
    join = lambda new_set: new_set & keys if keys is not None else new_set
    keys = join(text_files.keys())
    keys = join(image_files.keys())

    keys = list(sorted(keys))

    return keys, text_files, image_files


def get_image_dataset():
    """retrieve image dataset module without importing torch at the top level"""

    from torch.utils.data import Dataset

    class ImageDataset(Dataset):
        """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""

        def __init__(
            self,
            sample_processor,
            folder,
            media_file_extensions,
            input_sampler=lambda a: a,
        ):
            super().__init__()

            self.keys, text_files, media_files = folder_to_keys(
                folder, media_file_extensions
            )
            self.keys = input_sampler(self.keys)
            self.text_files = {k: v for k, v in text_files.items() if k in self.keys}
            self.media_files = {k: v for k, v in media_files.items() if k in self.keys}
            self.sample_processor = sample_processor

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, ind):
            key = self.keys[ind]
            output = {}

            media_file = self.media_files[key]
            data_tensor = self.sample_processor(media_file)
            output["data_tensor"] = data_tensor

            text_file = self.text_files[key]
            caption = text_file.read_text()
            output["text"] = caption

            return output

    return ImageDataset


def create_webdataset(
    urls,
    sample_processor,
    media_key="jpg",
    caption_key="txt",
    cache_path=None,
    input_sampler=lambda a: a,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    import webdataset as wds

    urls = input_sampler(urls)

    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10**10, handler=wds.handlers.warn_and_continue)

    def filter_dataset(item):
        if caption_key not in item:
            return False
        elif media_key not in item:
            return False
        else:
            return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}
        image_data = item[media_key]
        data_tensor = sample_processor(io.BytesIO(image_data))
        output["data_tensor"] = data_tensor

        text = item[caption_key]
        caption = text.decode("utf-8")
        output["text"] = caption

        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


class FilesReader:
    """FilesReader is a reader that reads files from a folder"""

    def __init__(
        self,
        sampler,
        sample_processor,
        input_dataset,
        media_file_extensions,
        batch_size,
        num_prepro_workers,
    ) -> None:
        super().__init__()
        dataset = get_image_dataset()(sample_processor, input_dataset, media_file_extensions, sampler)
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "files")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch


class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        sampler,
        sample_processor,
        input_dataset,
        batch_size,
        num_prepro_workers,
        wds_media_key="jpg",
        wds_caption_key="txt",
        cache_path=None,
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            input_dataset,
            sample_processor,
            media_key=wds_media_key,
            caption_key=wds_caption_key,
            cache_path=cache_path,
            input_sampler=sampler,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
