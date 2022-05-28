# ClipCap
Using pretrained encoder models (such as CLIP) & GPT-2 to generate captions from images. More details to come soon.

## Installation
```bash
pip install git+https://github.com/TheoCoombes/ClipCap.git
```

## Preprocessing Datasets
You can run the preprocess script using the command below - [More info](/docs/data_preprocessing.md).
```bash
python3 -m ClipCap.preprocess --help
```

## Training
You can train new models using preprocessed datasets using the `train.py` script:
```bash
python3 -m ClipCap.train --help
```

## TODO
Improved documentation and eval + inference scripts to come soon.