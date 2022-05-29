# ClipCap
Using pretrained encoder and language models to generate captions from multimedia inputs, allowing high fidelity text generation using the rich textual detail already learned by pretrained LMs on tasks such as image captioning, VQA, audio captioning and more.

More details and results to come soon.

## Installation
By default, the encoders remained uninstalled for ease of access. View the [data preprocessing](/docs/data_preprocessing.md) documentation for info on how to install these.
```bash
pip install git+https://github.com/TheoCoombes/ClipCap.git
```

## Supported Encoders
* [CLIP](https://github.com/openai/CLIP) for tasks such as Image Captioning, VQA etc.
* [CLAP](https://github.com/LAION-AI/CLAP) for tasks such as Audio Captioning, Audio Question Answering, etc.

## [Data Preprocessing](/docs/data_preprocessing.md)
You can run the data preprocess script using the command below. ([More info](/docs/data_preprocessing.md))
```bash
python3 -m clipcap.preprocess --help
```

## [Training](/docs/training.md)
You can run the training script using preprocessed data with the command below. ([More info](/docs/training.md))
```bash
python3 -m clipcap.train --help
```

## Acknowledgments
This repository is heavily based on [@rmokady](https://github.com/rmokady)'s [original implementation of ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and also contains modified versions of [@rom1504](https://github.com/rom1504)'s [clip-inference](https://github.com/rom1504/clip-retrieval/tree/76ac7c5cab2ca8e949f0bec479651baa58066684/clip_retrieval/clip_inference) and [embedding-reader](https://github.com/rom1504/embedding-reader) libraries. Many thanks to both for their amazing work :)

## TODO
Improved documentation and eval + inference scripts to come soon.