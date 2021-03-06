# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import scipy as sp
import scipy.io
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_dir', 'cifar10_test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, labels, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      total_pre = []
      total_tag = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      print('num_iter=', num_iter)
      while step < num_iter and not coord.should_stop():
        predictions, tags = sess.run([top_k_op, labels])
        #predictions = sess.run([top_k_op])
        total_pre.append(predictions)
        total_tag.append(tags)
        
        true_count += 1#np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      #print(' size of total_result=', len(total_result))
      precision = 0#true_count / total_sample_count
      #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      sp.io.savemat('test_fm_da', mdict={'predictions': total_pre, 'tags':total_tag})

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)



def eval_once_given_model(saver, summary_writer, logits, labels, summary_op, images):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  #loss = cifar10.loss(logits, labels)
  with tf.Session() as sess:
    #need to modify for test
    saver.restore(sess, "cifar10_train/model.ckpt-5000")
    global_step = 5000

    sess.run(tf.initialize_variables([v for v in tf.all_variables() if v.name.startswith("input_producer")]))
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      total_pre = []
      total_tag = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / 1))
      print('num_examples=', FLAGS.num_examples, '  batch_size=', 1, '  num_iter=', num_iter)
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      #images_value, predictions, tags, localvalue = sess.run([images, top_k_op, labels, local])

      #for v in tf.trainable_variables():
      #  print(v.name, ':', sess.run(v))

      
      while step < num_iter and not coord.should_stop():
        images_value, predictions, tags = sess.run([images, logits, labels])
        #predictions = sess.run([top_k_op])
        total_pre.append(predictions)
        total_tag.append(tags)
        #print('step:', step)
        true_count += 1#np.sum(predictions)
        step += 1
      


      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      #summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
      print('get a exception')

    finally:
      print('finally')
      np.savez('cnn-da', y_pred=total_pre, y_true=total_tag)
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=100)

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, labels = cifar10.inputs(eval_data='test')
    # need modify for test
    logits, _ = cifar10.inference(images)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    del variables_to_restore['input_producer/limit_epochs/epochs']
    saver = tf.train.Saver(variables_to_restore)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.test_dir, g)

    while True:
      eval_once_given_model(saver, summary_writer, logits, labels, summary_op, images)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.test_dir):
    tf.gfile.DeleteRecursively(FLAGS.test_dir)
  tf.gfile.MakeDirs(FLAGS.test_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
