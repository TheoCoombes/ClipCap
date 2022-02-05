from pathlib import Path
from tqdm import tqdm
import shutil
import fire

def merge_datasets(
    base_dir: str = "../datasets/",
    out_dir: str = "../final_dataset/",
    move_files: bool = False, # move rather than copy
):
    path = Path(base_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    out_images = output_path / "img_embeddings"
    out_tokens = output_path / "text_tokens"
    out_masks = output_path / "text_masks"

    out_images.mkdir(exist_ok=True)
    out_tokens.mkdir(exist_ok=True)
    out_masks.mkdir(exist_ok=True)

    for folder in tqdm([x.resolve() for x in path.iterdir() if x.is_dir()], desc='Merging Datasets'):
        images_path = folder / "img_embeddings"
        tokens_path = folder / "text_tokens"
        masks_path = folder / "text_masks"

        images = {file.name.split("_")[-1]: file for file in images_path.glob("*.npy")}
        tokens = {file.name.split("_")[-1]: file for file in tokens_path.glob("*.npy")}
        masks = {file.name.split("_")[-1]: file for file in masks_path.glob("*.npy")}

        for filename, image_file in images.items():
            tokens_file, masks_file = tokens.get(filename, None), masks.get(filename, None)

            if (tokens is None) or (masks is None):
                # TODO warn?
                continue

            if move_files:
                f = shutil.move
            else:
                f = shutil.copy
            
            f(image_file, out_images / f"{folder.name}-{filename}")
            f(tokens_file, out_images / f"{folder.name}-{filename}")
            f(masks_file, out_images / f"{folder.name}-{filename}")
    
    print("done.")


if __name__ == '__main__':
    fire.Fire(merge_datasets)