## Theme: Speech_Emotion_Classifier

Made by inspiring https://github.com/asarsembayev/audio_classifier
Used Conv2d NN-architecture in PyTorch library

Accuracy on test: 83%

Used dataset: RAVDESS

## Installation

Please install necessary libraries with
```
pip install -r requirements.txt
```

## Usage
```
usage: main.py [-h] [--transforms TRANSFORMS]
               [--configs_path CONFIGS_PATH]
               src_dir labels_path

positional arguments:
  src_dir               [input] path to directory with raw audio files
  labels_path           [input] path to csv w/ labels

optional arguments:
  -h, --help            show this help message and exit
  --transforms TRANSFORMS
                        [input] transforms
  --configs_path CONFIGS_PATH
                        [input] configs_path
```
