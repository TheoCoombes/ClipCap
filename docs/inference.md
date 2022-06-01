# Inference

## Prerequisites
You must have a pretrained model saved locally, as well as the associated config (`.yaml`) file.

## Run Inference Demo
You can easily run an inference demo over the command line using the following command.
```bash
python -m clipcap.inference --help
```
Running the command with `--help` provides extensive documentation for each individual argument.

## Using ClipCap Programmatically
```py
import clipcap, torch

device = "cuda:0"

model, tokenizer = clipcap.load("./model.pt", "./model_config.yaml", device=device)
encode_fn, preprocess = clipcap.get_encoder_from_model(model, device=device)

sample = preprocess("./image.jpg").unsqueeze(0).to(device)

with torch.no_grad():
    embedding = encode_fn(sample)
    embedding_prefix = model.transformer_mapper(embedding)

# Or, alternatively use your own GPT-2 (or so) based inference algorithm. Please submit a PR if you do so :)
captions = clipcap.inference.base.generate_beam(
    model, tokenizer, embedding_prefix
)

print(captions)
```


## Example Usage
Example for training a model using a preprocessed clotho dataset (see [data_preprocessing.md](/docs/data_preprocessing.md)) on gpu 0 with the huggingface model `gpt2` (gpt2 small) for 25 epochs with a batch size of 32. The rest are default settings.
```bash
python3.8 -m clipcap.train \
    --input-dataset "./preprocessed/clotho/train/"
    --output-folder "./trained/clotho_initial_tests/"
    --epochs 25
    --checkpoint-filename-prefix "clotho_initial_tests"
    --device "0"
    --language-model "gpt2"
    --batch-size 32
```