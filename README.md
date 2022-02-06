# CLIP-Image-Captioning
[WIP] Using CLIP+GPT2 to generate captions from images.

## Preprocessing Datasets
You can preprocess existing datasets using the `create_dataset.py` script:
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

## Training
You can train new models using preprocessed datasets using the `train.py` script:
```bash
python3 train.py \
    --data_dir "./preprocessed_dataset/"
    --output_dir "./preprocessed_dataset/"
    --output_filename_prefix "demo_model" # output for the model files, e.g. 'demo_model_latest.ckpt'
    --epochs 5
    --save_every_epochs 1 # save the model every x epochs, e.g. 'demo_model_0.ckpt'
    --save_every_steps 10000 # save the model every x steps, updating the existing file 'demo_model_latest.ckpt'
    --prefix_length 40
    --prefix_size 512 # the output shape of the CLIP embeddings
    --mapping_type "transformer" # the projection mapping type, either 'transformer' or 'mlp'
    --only_prefix # trains only the transformer/mlp layer, and does not finetune the language model
    --num_layers 8 # the number of layers for the mapping type
    --num
    --normalize_prefix # normalizes the clip embeddings for training
    --use_deepspeed # [EXPERIMENTAL] uses deepspeed for training (for more than 1 gpu)
    --gpu_devices "0" # sets the training GPU device(s) to use. "<number>" / "<number>,<number>,..." / "-1" for all
    **huggingface_kwargs # to provide kwargs for huggingface downloads etc.
```
All of the trainer's arguments can be found at the [train(...) method](https://github.com/TheoCoombes/CLIP-Image-Captioning/blob/main/train.py#L36).
