#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as learn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

tf.logging.set_verbosity(tf.logging.INFO)

TIMESERIES_COL = 'rawdata'
N_OUTPUTS = 1  # in each sequence, 1-6 are features, and 7 is label
SEQ_LEN = None
DEFAULTS = None
N_INPUTS = None

def init(hparams):
  global SEQ_LEN, DEFAULTS, N_INPUTS
  SEQ_LEN =  hparams['sequence_length']
  DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
  N_INPUTS = SEQ_LEN - N_OUTPUTS

# read data and convert to needed format
def read_dataset(filename, mode, batch_size):
  def _input_fn():
    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=None, shuffle=True)

    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)

    value_column = tf.expand_dims(value, -1)
    #print ('readcsv={}'.format(value_column))

    # all_data is a list of tensors
    all_data = tf.decode_csv(value_column, record_defaults=DEFAULTS)
    inputs = all_data[:len(all_data)-N_OUTPUTS]  # first few values
    label = all_data[len(all_data)-N_OUTPUTS : ] # last few values

    # from list of tensors to tensor with one more dimension
    inputs = tf.concat(inputs, axis=1)
    label = tf.concat(label, axis=1)
    print ('inputs={}'.format(inputs))

    return {TIMESERIES_COL: inputs}, label   # dict of features, label
  return _input_fn


def lstm_model(features, mode, params):
  #LSTM_SIZE = N_INPUTS//3  # size of the internal state in each of the cells
  LSTM_SIZE = 6
  # 1. dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
  x = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1])

  # 2. configure the RNN
  lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
  outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
  outputs = outputs[:, (N_INPUTS-1):, :]  # last cell only

  # 3. flatten lstm output and pass through a dense layer
  lstm_flat = tf.reshape(outputs, [-1, lstm_cell.output_size])
  h1 = tf.layers.dense(lstm_flat, N_INPUTS//2, activation=tf.nn.relu)
  predictions = tf.layers.dense(h1, 1, activation=None) # (?, 1)
  return predictions

def serving_input_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }

    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }
    features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2])

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
