# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
from tensorflow import flags
from scipy.io import wavfile

FLAGS = flags.FLAGS

flags.DEFINE_integer("first_class", 276, "First class of the dataset to evaluate.")
flags.DEFINE_integer("second_class", 282, "Second class of the dataset to evaluate.")
  
parser = argparse.ArgumentParser(description='Read file and process audio')
parser.add_argument('wav_file', type=str, help='File to read and process.')
parser.add_argument('--class_labels', type=str, help='Class labels to predict.')
parser.add_argument('--to_csv', type=bool, default=False, metavar='CSV', help='Predictions to csv file.')
parser.add_argument('--ten_seconds', type=bool, default=False, metavar='LIMIT_SECONDS', help='A label for each 10 seconds of the wav.')
parser.add_argument('--num_predictions', type=int, default=7, metavar='PREDICTIONS', help='Number of predictions.')
parser.add_argument('--threshold', type=float, default=0.1, metavar='THRESHOLD', help='Threshold to discard tags.')


def process_file(wav_file, class_labels, to_csv, ten_seconds, num_predictions, threshold):
    sr, data = wavfile.read(wav_file)
    if data.dtype != np.int16:
        raise TypeError('Bad sample type: %r' % data.dtype)

    # local import to reduce start-up time
    from audio.processor import WavProcessor, format_predictions

    with WavProcessor() as proc:
        print('Total predicciones ' + str(num_predictions))
        print('Umbral de corte ' + str(threshold))
        if ten_seconds == False:
          predictions = proc.get_predictions(sr, data, num_predictions, threshold, class_labels)
          print('Predictions')
          #print(predictions)
          print(format_predictions(predictions))
			
        else:
          predictions = proc.get_predictions2(sr, data, num_predictions, threshold, class_labels)
          print('Predictions')
          for i in range(0, len(predictions)):
            #print(predictions[i])
            print(str(i)+' '+format_predictions(predictions[i]))

    


if __name__ == '__main__':
    args = parser.parse_args()
    process_file(**vars(args))
