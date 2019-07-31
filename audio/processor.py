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

import csv
import os
import numpy as np
import tensorflow as tf

from . import params
from .utils import vggish, youtube8m


__all__ = ['WavProcessor', 'format_predictions']


cwd = os.path.dirname(os.path.realpath(__file__))


def format_predictions(predictions):
    return ', '.join('{0}: {1:.2f}'.format(*p) for p in predictions)


class WavProcessor(object):
    _class_map = {}
    _vggish_sess = None
    _youtube_sess = None

    def __init__(self):
        pca_params = np.load(params.VGGISH_PCA_PARAMS)
        self._pca_matrix = pca_params[params.PCA_EIGEN_VECTORS_NAME]
        self._pca_means = pca_params[params.PCA_MEANS_NAME].reshape(-1, 1)

        self._init_vggish()
        self._init_youtube()
        self._init_class_map()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self._vggish_sess:
            self._vggish_sess.close()

        if self._youtube_sess:
            self._youtube_sess.close()

    def _init_vggish(self):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            vggish.model.define_vggish_slim(training=False)
            vggish.model.load_vggish_slim_checkpoint(sess, params.VGGISH_MODEL)

        self._vggish_sess = sess

    def _init_youtube(self):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            youtube8m.model.load_model(sess, params.YOUTUBE_CHECKPOINT_FILE)

        self._youtube_sess = sess

    def _init_class_map(self):
        with open(params.CLASS_LABELS_INDICES) as f:
            next(f)  # skip header
            reader = csv.reader(f)
            for row in reader:
                self._class_map[int(row[0])] = row[2]
				
    def get_predictions(self, wav_file, sample_rate, data, num_predictions, threshold, class_labels):
        samples = data / 32768.0  # Convert to [-1.0, +1.0]
		
        examples_batch = vggish.input.waveform_to_examples(samples, sample_rate)
        features = self._get_features(examples_batch)
        predictions = self._process_features(features)
        #print(predictions)
        #print(predictions[0])
        predictions = self._filter_predictions(predictions, num_predictions, threshold, class_labels)
		
        #features2 = self._toCSV(wav_file, predictions)
		
        return predictions
		
    def get_predictions2(self, sample_rate, data, num_predictions, threshold, class_labels):
        samples = data / 32768.0  # Convert to [-1.0, +1.0]
		
        num_examples = len(samples) / 441000
        num_10s = 44100
        total_predictions = []
        for i in range(0,int(num_examples)):
          #print(int(num_examples/10)+1)
          #print(len(total_predictions))
          num_10s = 441000*(i+1)
          samples_10seconds = samples[441000*i:num_10s]
          examples_batch = vggish.input.waveform_to_examples(samples_10seconds, sample_rate)
          features = self._get_features(examples_batch)
          predictions = self._process_features(features)
          #print(predictions)
          #print(predictions[0])
          predictions = self._filter_predictions(predictions, num_predictions, threshold, class_labels)
          total_predictions.append(predictions)
          #print(total_predictions[i])
          if i == int(num_examples)-1:
            #print(total_predictions[i])
            samples_10seconds = samples[num_10s:len(samples)]
            examples_batch = vggish.input.waveform_to_examples(samples_10seconds, sample_rate)
            features = self._get_features(examples_batch)
            predictions = self._process_features(features)
            predictions = self._filter_predictions(predictions, num_predictions, threshold, class_labels)
            total_predictions.append(predictions)
		
        return total_predictions
		
		
    def _filter_predictions(self, predictions, num_predictions, threshold, class_labels):
        count = num_predictions
        hit = threshold

        top_indices = np.argpartition(predictions[0], -count)[-count:]
        labels  = class_labels.split(",")
        top_mood = np.zeros(shape=(len(labels)), dtype=int)
        for k in range(0, len(labels)):
          top_mood[k] = labels[k]
        #print(predictions)
		
        total_mood = 0
        for j in range(top_mood[0], top_mood[len(top_mood)-1]):
          total_mood = total_mood + predictions[0][j]
        for j in range(top_mood[0], top_mood[len(top_mood)-1]):
          predictions[0][j] = predictions[0][j] / total_mood
		
        line = ((self._class_map[i], float(predictions[0][i])) for
                i in top_mood if predictions[0][i] > hit)
        #print(hit)
        return sorted(line, key=lambda p: -p[1])

    def _process_features(self, features):
        sess = self._youtube_sess
        num_frames = np.minimum(features.shape[0], params.MAX_FRAMES)
        data = youtube8m.input.resize(features, 0, params.MAX_FRAMES)
        data = np.expand_dims(data, 0)
        num_frames = np.expand_dims(num_frames, 0)

        input_tensor = sess.graph.get_collection("input_batch_raw")[0]
        num_frames_tensor = sess.graph.get_collection("num_frames")[0]
        predictions_tensor = sess.graph.get_collection("predictions")[0]
        #print(data)
        #print(input_tensor)
        #print(predictions_tensor)

        predictions_val, = sess.run(
            [predictions_tensor],
            feed_dict={
                input_tensor: data,
                num_frames_tensor: num_frames
            })

        return predictions_val

    def _get_features(self, examples_batch):
        sess = self._vggish_sess
        features_tensor = sess.graph.get_tensor_by_name(
            params.VGGISH_INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            params.VGGISH_OUTPUT_TENSOR_NAME)

        [embedding_batch] = sess.run(
            [embedding_tensor],
            feed_dict={features_tensor: examples_batch}
        )

        postprocessed_batch = np.dot(
            self._pca_matrix, (embedding_batch.T - self._pca_means)
        ).T

        return postprocessed_batch
		
		
    def toCSV(self, data, wav_file, predictions):
        num_examples = int(len(data) / 441000)
        file_name = os.path.basename(wav_file)
        with open('predictions.csv', mode='w') as predictions_file:
            prediction_writer = csv.writer(predictions_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            prediction_writer.writerow(['Wav name', 'Start seconds', 'End seconds', 'Prediction'])
            prediction_writer.writerow([file_name, 0, num_examples, predictions])

    def toCSV2(self, data, wav_file, predictions):
        num_examples = int(len(data) / 44100)
        file_name = os.path.basename(wav_file)
        with open('predictions.csv', mode='w') as predictions_file:
            prediction_writer = csv.writer(predictions_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            prediction_writer.writerow(['Wav name', 'Start seconds', 'End seconds', 'Prediction'])
            for i in range(0,len(predictions)):
                start = i*10
                if i < len(predictions)-1:
                  end = (i+1)*10
                else:
                  end = num_examples-start
                print(start)
                print(end)
                print(num_examples)
                prediction_writer.writerow([file_name, start, end, format_predictions(predictions[i])])
            