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
from tensorflow import gfile
from tensorflow import flags
from scipy.io import wavfile
  
parser = argparse.ArgumentParser(description='Read file and process audio')
parser.add_argument('wav_file', type=str, help='File to read and process.')
parser.add_argument('--class_labels', type=str, default='276,277,278,279,280,281,282', help='Class labels to predict.')
parser.add_argument('--to_csv', type=bool, default=False, metavar='CSV', help='Predictions to csv file.')
parser.add_argument('--output_file', type=str, default='predictions.csv', help='The file to save the predictions to.')
parser.add_argument('--ten_seconds', type=bool, default=False, metavar='LIMIT_SECONDS', help='A label for each 10 seconds of the wav.')
parser.add_argument('--num_predictions', type=int, default=7, metavar='PREDICTIONS', help='Number of predictions.')
parser.add_argument('--threshold', type=float, default=0.1, metavar='THRESHOLD', help='Threshold to discard tags.')


def process_file(wav_file, class_labels, to_csv, output_file, ten_seconds, num_predictions, threshold):
    files = gfile.Glob(wav_file)
    print(len(files))
    if not files:
        raise IOError("Unable to find input files. data_pattern='" +wav_file + "'")
    logging.info("number of input files: " + str(len(files)))
    sr, data = wavfile.read(wav_file)
    sr, data = wavfile.read(wav_file)
    if data.dtype != np.int16:
        raise TypeError('Bad sample type: %r' % data.dtype)

    # local import to reduce start-up time
    from audio.processor import WavProcessor, format_predictions

    with WavProcessor() as proc:
        print('Total predicciones ' + str(num_predictions))
        print('Umbral de corte ' + str(threshold))
        if ten_seconds == False:
          predictions = proc.get_predictions(wav_file, sr, data, num_predictions, threshold, class_labels)
          print('Predictions')
          #print(predictions)
          print(format_predictions(predictions))
          if to_csv == True:
            proc.toCSV(data, wav_file, output_file, format_predictions(predictions))
          
			
        else:
          predictions = proc.get_predictions2(sr, data, num_predictions, threshold, class_labels)
          print('Predictions')
          for i in range(0, len(predictions)):
            #print(predictions[i])
            print(str(i)+' '+format_predictions(predictions[i]))
          if to_csv == True:
            proc.toCSV2(data, wav_file, output_file, predictions)
            

    


if __name__ == '__main__':
    args = parser.parse_args()
    process_file(**vars(args))
