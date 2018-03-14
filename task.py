import traceback
import argparse
import json
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

def compute_errors(features, labels, predictions):
   if predictions.shape[1] == 1:
      loss = tf.losses.mean_squared_error(labels, predictions)
      rmse = tf.metrics.root_mean_squared_error(labels, predictions)
      accuracy = tf.metrics.accuracy(labels, tf.floor(predictions))
      return loss, rmse, accuracy
   else:
      # one prediction for every input in sequence
      # get 1-N of (x + label)
      labelsN = tf.concat([features[model.TIMESERIES_COL], labels], axis=1)
      labelsN = labelsN[:, 1:]
      # loss is computed from the last 1/3 of the series
      N = (2 * model.N_INPUTS) // 3
      loss = tf.losses.mean_squared_error(labelsN[:, N:], predictions[:, N:])
      # rmse is computed from last prediction and last label
      lastPred = predictions[:, -1]
      rmse = tf.metrics.root_mean_squared_error(labels, lastPred)
      return loss, rmse

# create the inference model
def sequence_regressor(features, labels, mode, params):

  # 1. run the model
  model_func = getattr(model, 'lstm_model')
  predictions = model_func(features, mode, params)

  # 2. loss function, training/eval ops
  loss = None
  train_op = None
  eval_metric_ops = None
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
     loss, rmse, accuracy = compute_errors(features, labels, predictions)

     if mode == tf.estimator.ModeKeys.TRAIN:
        # this is needed for batch normalization, but has no effect otherwise
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           # 2b. set up training operation
           train_op = tf.contrib.layers.optimize_loss(
              loss,
              tf.train.get_global_step(),
              learning_rate=params['learning_rate'],
              optimizer="Adam")

     # 2c. eval metric
     eval_metric_ops = {
      "rmse": rmse,
      "accuracy": accuracy
     }

  # 3. Create predictions
  if predictions.shape[1] != 1:
     predictions = predictions[:, -1] # last predicted value
  predictions_dict = {"predicted": predictions}

  # 4. return EstimatorSpec
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs={'predictions': tf.estimator.export.PredictOutput(predictions_dict)}
  )



def create_custom_estimator(output_dir, hparams):
  save_freq = max(1, min(100, hparams['train_steps']/100))
  training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None,
                                               save_checkpoints_steps=save_freq)
  return tf.estimator.Estimator(model_fn=sequence_regressor, model_dir=output_dir,
                                config=training_config, params=hparams)


def generate_experiment_fn(output_dir, hparams):
 def experiment_fn(output_dir):
    get_train = model.read_dataset(hparams['train_data_paths'],
                                   tf.estimator.ModeKeys.TRAIN,
                                   hparams['train_batch_size'])
    get_valid = model.read_dataset(hparams['eval_data_paths'],
                                   tf.estimator.ModeKeys.EVAL,
                                   1000)
    eval_freq = max(1, min(2000, hparams['train_steps']/5))

    return tf.contrib.learn.Experiment(
        estimator=create_custom_estimator(output_dir, hparams),
        train_input_fn=get_train,
        eval_input_fn=get_valid,
        train_steps=hparams['train_steps'],
        eval_steps=1,
        min_eval_frequency=eval_freq,
        export_strategies=[saved_model_export_utils.make_export_strategy(
            model.serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )]
    )
 return experiment_fn

if __name__ == '__main__':
  hparams = {'train_data_paths': 'readyData.csv',
             'eval_data_paths': 'readyData.csv',
             'train_batch_size': 5,
             'learning_rate': 0.01,
             'train_steps': 100,
             'sequence_length': 6,
             'eval_delay_secs': 10,
             'min_eval_frequency': 1}

  output_dir = ('output')

  model.init(hparams)

  # Run the training job
  try:
     learn_runner.run(generate_experiment_fn(output_dir, hparams), output_dir)
  except:
     traceback.print_exc()
