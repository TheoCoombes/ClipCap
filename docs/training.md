# Training

## Prerequisites
You must have a preprocessed dataset in order to begin training. View the [data preprocessing documentation](/docs/data_preprocessing.md) for more info. 

## Start Training
Once you have preprocessed your dataset, you can run start a training run using the command below:
```bash
python -m clipcap.train --help
```

Running the command with `--help` provides extensive documentation for each individual argument. Alternatively, you can view the [train/args.py file](/clipcap/train/args.py). However note that some arguments may be found in some other files, such as the [model/args.py file](/clipcap/model/args.py).

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