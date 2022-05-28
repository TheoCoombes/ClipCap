# Preprocessing Data
Both the preprocessor and dataloaders are modified versions of the [clip-inference](https://github.com/rom1504/clip-retrieval/tree/76ac7c5cab2ca8e949f0bec479651baa58066684/clip_retrieval/clip_inference) and [embedding-reader](https://github.com/rom1504/embedding-reader) libraries respectively. Many thanks to [@rom1504](https://github.com/rom1504) for his amazing work :)

## Prerequisites
By default, the encoders remained uninstalled for ease of access. You can install the neccesary libraries for each encoder using the commands shown below.

### CLIP Encoder (Images)
CLIP requires both the `PIL` library and the `clip` library installed. You can install the latest version of `clip` using the command below:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### CLAP Encoder (Audio)
CLAP requires the `clap` branch of the LAION fork of `open_clip` installed. It can be installed like so:
```bash
pip install git+https://github.com/LAION-AI/CLAP.git@clap
```
CLAP also requires `torchaudio` installed for the audio transforms, also installable using the command below:
```bash
pip install torchaudio
```

### Other Encoders
Feel free to submit and issue and/or PR to help support other encoders. This repo has been designed to be as modular and reproducible as possible, and it has therefore been made very easy to implement your own encoders into ClipCap. You can take a look at the [sample encoder script](/ClipCap/encoders/_baseformat.py) as a guide when creating your pull requests.

## Running the Preprocessor
Once all requirements (including `ClipCap`) have been installed, you can run the preprocessor using the command below:
```bash
python -m ClipCap.preprocess --help
```

Running the command with `--help` provides extensive documentation for each individual argument. Alternatively, you can view the [preprocess/args.py file](/ClipCap/preprocess/args.py).

## Example
```bash
TODO
```