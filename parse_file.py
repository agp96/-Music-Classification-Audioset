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
parser.add_argument('wav_file', type=str, help='File to read and process')


def process_file(wav_file):
    sr, data = wavfile.read(wav_file)
    if data.dtype != np.int16:
        raise TypeError('Bad sample type: %r' % data.dtype)

    # local import to reduce start-up time
    from audio.processor import WavProcessor, format_predictions

    with WavProcessor() as proc:
        predictions = proc.get_predictions(sr, data, FLAGS.first_class, FLAGS.second_class)

    print(format_predictions(predictions))


if __name__ == '__main__':
    args = parser.parse_args()
    process_file(**vars(args))
