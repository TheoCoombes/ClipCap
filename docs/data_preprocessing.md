# Preprocessing Data
Both the preprocessor and dataloaders are modified versions of the [clip-inference](https://github.com/rom1504/clip-retrieval/tree/76ac7c5cab2ca8e949f0bec479651baa58066684/clip_retrieval/clip_inference) and [embedding-reader](https://github.com/rom1504/embedding-reader) libraries respectively. Many thanks to [@rom1504](https://github.com/rom1504) for his amazing work :)

## Prerequisites
By default, the encoders remained uninstalled for ease of access. You can install the neccesary libraries for each encoder using the commands shown below.

### CLIP Encoder (Images)
CLIP requires both the `PIL` library and the `clip` library installed. You can install the latest version of `clip` using the command below:
```bash
pip install git+https://github.com/openai/CLIP.git
```
You can install all CLIP requirements in one command using `pip install -r requirements-clip.txt`.

### CLAP Encoder (Audio)
CLAP requires the `clap` branch of the LAION fork of `open_clip` installed. It can be installed like so:
```bash
pip install git+https://github.com/LAION-AI/CLAP.git@clap
```
CLAP also requires a few audio libraries to be installed for the audio transforms, also installable using the command below:
```bash
pip install torchaudio h5py torchlibrosa PySoundFile
```
You can install all CLAP requirements in one command using `pip install -r requirements-clap.txt`.

### Custom Encoders
Feel free to submit and issue and/or PR to help ClipCap easily support other encoders. This repo has been designed to be as modular and reproducible as possible, and it has therefore been made very easy to implement your own encoders into ClipCap.

You can take a look at the [sample encoder script](/clipcap/encoders/_baseformat.py) as a guide when creating your pull requests.

## Running the Preprocessor
Once all requirements (including `clipcap`) have been installed, you can run the preprocessor using the command below:
```bash
python -m clipcap.preprocess --help
```

Running the command with `--help` provides extensive documentation for each individual argument. Alternatively, you can view the [preprocess/args.py](/clipcap/preprocess/args.py) and [encoder/args.py](/clipcap/encoder/args.py) files.

## Example Usage
Example for preprocessing a webdataset with a pretrained CLAP model, with the webdataset containing FLAC audio files at key `flac` and captions of key `text` contained inside of json files inside the webdataset with the key `json`, resulting in a final `wds-caption-key` of `json/text`.
```bash
python3.8 -m clipcap.preprocess \
    --input-dataset "./datasets/clotho/train/{0..7}.tar"
    --output-folder "./preprocessed/clotho/train/"
    --input-format "webdataset"
    --batch-size 256
    --device "cuda:0"
    --write-batch-size 10000
    --wds-media-key "flac"
    --wds-caption-key "json/text"
    --wds-samples-per-file 512
    --encoder-model-name "clap"
    --encoder-model-variant "./clap/pretrained.pt"
```
