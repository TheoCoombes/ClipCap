# CLIP-Image-Captioning
[WIP] Using CLIP+GPT2 to generate captions from images.

## Preprocessing Datasets
The preprocessor allows for both webdataset and plain file datasets, and can be executed like so:
```bash
python3 create_dataset.py \
    --input_dataset "./webdataset/{000..123}.tar"
    --output_folder "./preprocessed_dataset/"
    --input_format "webdataset"
    --batch_size 1024
    --num_prepro_workers 16
    --wds_image_key "jpg"               # (required if input_format='webdataset') the webdataset key for the image files
    --wds_caption_key "txt"             # (required if input_format='webdataset') the webdataset key for the captions
    --clip_model "ViT-B/32"
    --tokenizer_model_type "gpt2"       # 'gpt2' / 'gpt-j' / 't5' / 't0'
    --tokenizer_model_variant "gpt2-xl" # the huggingface model name
    --max_token_length 128              # captions with a token length > max_token_length will be truncated
    --prefix_length 10                  # the length for the projected prefixes (must be the same for the training run)
    --device "cuda:0"                   # the cuda device to be used
```
All of the preprocessor's arguments can be found at the [clip_inference(...) method](https://github.com/TheoCoombes/CLIP-Image-Captioning/blob/main/create_dataset.py#L317).
