[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)

# Music Clasification Audioset
Audio classification feature demo\
Detailed description can be found [here](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/)

## Installation
* Get a copy of this repo
* Install system packages
```bash
sudo apt-get install libportaudio2 portaudio19-dev
```
* Install python requirements
```bash
pip install -r requirements.txt
```

* Download and extract saved models to source directory
```bash
wget https://s3.amazonaws.com/audioanalysis/models.tar.gz
tar -xzf models.tar.gz
```

## Running
#### To process prerecorded wav file
run
```bash
python parse_file.py path_to_your_file.wav
```
_Note: file should have 16000 rate_

#### Features added

If you want to choose which labels you want to be used to predict, you can add --class_labels, for example --class_label='0,137' to predict Speech or Music. 

You can also change the number of predictions you want to display, --num_predictions with 7 as default, and the threshold, -threshold with 0.1 as default.

Also the audio file can be divided into 10 seconds, obtaining a label for each segment, --ten_seconds=True to activate it.

The results of the predictions will be displayed on the console, and for convenience they can be exported to csv. With --to_csv=True to active it and if you want to save in a specific file with --output_file.

Example with the previous features added.

```bash
python parse_file.py "../gdrive/My Drive/example.wav" --class_labels='0,137' --num_predictions=2 --threshold=0.5 --ten_seconds=True --to_csv=True --output_file='../gdrive/My Drive/predictions.csv' 
```

In case of a large number of files to predict, a number of audios can be bounded with --num_files, the default value is 0 which means that it analyzes all files.

```bash
python parse_file.py "../gdrive/My Drive/PC_VideoGame_Music/*.wav" --class_labels='276,277,278,279,280,281,282' --num_predictions=7 --threshold=0.15 --ten_seconds=True --to_csv=True --output_file='../gdrive/My Drive/predictions.csv' --num_files=500
```

#### To capture and process audio from mic
run
```bash
python capture.py
```
It will capture and process samples in a loop.\
To get info about parameters run
```bash
python capture.py --help
```

#### To start web server
run
```bash
python daemon.py
```
By default you can reach it on http://127.0.0.1:8000 \
It will:
* Capture data form your mic
* Process data
* Send predictions to web interface
* Send predictions to devicehive

Also you can configure your devicehive connection though this web interface.

## Useful info
To train classification model next resources have been used:
* [Google AudioSet](https://research.google.com/audioset/)
* [YouTube-8M model](https://github.com/google/youtube-8m)
* [Tensorflow vggish model](https://github.com/tensorflow/models/tree/master/research/audioset)

You can try to train model with more steps/samples to get more accuracy.
