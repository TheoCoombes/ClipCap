# ClipCap
Using pretrained encoder and language models to generate captions from multimedia inputs, allowing high fidelity text generation using the rich detail learned from pretrained LMs on tasks such as image captioning, VQA, audio captioning and more. More details and results to come soon.

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

## Acknowledgments
This repository is heavily based on [@rmokady](https://github.com/rmokady)'s [original implementation of ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and also contains modified versions of [@rom1504](https://github.com/rom1504)'s [clip-inference](https://github.com/rom1504/clip-retrieval/tree/76ac7c5cab2ca8e949f0bec479651baa58066684/clip_retrieval/clip_inference) and [embedding-reader](https://github.com/rom1504/embedding-reader) libraries. Many thanks to both for their amazing work :)

## TODO
Improved documentation and eval + inference scripts to come soon.