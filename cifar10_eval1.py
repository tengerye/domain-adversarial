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
from multiprocessing import Pool

import numpy as np
import scipy as sp
import scipy.io
import tensorflow as tf

import cifar10
import input

IMAGE_SIZE = 112

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train',
                           #"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")




def eval_once_given_model(saver, loss, loadfile, num_examples):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    #need to modify for test
    saver.restore(sess, loadfile) 
    global_step = int(loadfile.split('-')[1])

    # Create placeholder.
    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))
    domain_labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))

    loss_label = tower_loss_label(images, labels)
    loss_domain = tower_loss_domain(images, domain_labels)

    #sess.run(tf.initialize_variables([v for v in tf.all_variables() if v.name.startswith("input_producer")]))

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      step = 0 
      while step < 3 and not coord.should_stop():
        #images_value, predictions, tags = sess.run([images, top_k_op, labels])
        label_loss_value, domain_loss_value = sess.run([loss_label, loss_domain], feed_dict = {images:x_train, labels:y_train, domain_labels:domain_train})
        step += 1
      

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    finally:
      #print('finally')
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=100)


def test_once_given_model(saver, loadfile, x, y, domain):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    #need to modify for test
    saver.restore(sess, loadfile) 
    global_step = int(loadfile.split('-')[1])

    # Create placeholder.
    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))
    domain_labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))

    loss_label = tower_loss_label(images, labels)
    loss_domain = tower_loss_domain(images, domain_labels)

    #sess.run(tf.initialize_variables([v for v in tf.all_variables() if v.name.startswith("input_producer")]))

    # Start the queue runners.
    return sess.run([loss_label, loss_domain], feed_dict = {images:x, labels:y, domain_labels:domain})
      

 

def tower_loss_label(images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  tf.get_variable_scope().reuse_variables()

  # Build inference Graph.
  logits, _ = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will

  return cifar10.loss_without_decay(logits, labels)



def tower_loss_domain(images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  tf.get_variable_scope().reuse_variables()

  # Build inference Graph.
  _, logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will

  return cifar10.loss_without_decay(logits, labels)


def calculate_loss(images, labels, domains):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """


  logits1, logits2 = cifar10.inference(images)


  # Build the portion of the Graph calculating the losses. Note that we will

  return cifar10.loss_without_decay(logits1, labels), \
          cifar10.loss_without_decay(logits2, labels)


def calculate_inference(images, labels, domains):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """


  return cifar10.inference(images)


# Given a data set, return the evaluation (cross entropy) result of it.
def eval_a_dataset(saver, filename, x, y, domain, images, labels, domain_labels, loss_label, loss_domain):

    # Supplement the number of examples to multiplier of 25.
    num_of_examples = np.shape(x)[0]

    #remainder = FLAGS.batch_size - int(math.ceil(num_of_examples/FLAGS.batch_size))
    #index = range(num_of_examples) + [0] * remainder


    with tf.Session() as sess:
       # Load save files.
       saver.restore(sess, filename) 
       global_step = int(filename.split('-')[1])

       # Allocate results in a list.
       losses_label = []
       losses_domain = []

       # Start the queue runners.
       step = 0
       #while step + FLAGS.batch_size <= len(index):
           
       while step + FLAGS.batch_size <= num_of_examples:
           label_loss_value, domain_loss_value = sess.run([loss_label, loss_domain], feed_dict = {images:x[step:step+FLAGS.batch_size, :], labels:y[step:step+FLAGS.batch_size], domain_labels:domain[step:step+FLAGS.batch_size]})

           losses_label.append(label_loss_value)
           losses_domain.append(domain_loss_value)
           step = step + FLAGS.batch_size

    return np.mean(losses_label), np.mean(losses_domain)


# Given a data set, return the actual prediction of it.
def predict_a_dataset(saver, filename, x, y, domain, images, labels, domain_labels, loss_label, loss_domain):

    # Supplement the number of examples to multiplier of 25.
    num_of_examples = np.shape(x)[0]

    # Batch prediction.
    with tf.Session() as sess:
       # Load save files.
       saver.restore(sess, filename) 
       global_step = int(filename.split('-')[1])

       # Allocate results in a list.
       losses_label = []
       losses_domain = []

       # Start the queue runners.
       step = 0
           
       while step + FLAGS.batch_size <= num_of_examples:
           label_loss_value, domain_loss_value = sess.run([loss_label, loss_domain], feed_dict = {images:x[step:step+FLAGS.batch_size, :]})

           losses_label.append(label_loss_value)
           losses_domain.append(domain_loss_value)
           step = step + FLAGS.batch_size

       # change the shape of result.
       losses_label = np.asarray(losses_label).reshape((-1, 21))
       losses_domain = np.asarray(losses_domain).reshape((-1, 2))

    return np.argmax(losses_label, axis=1), np.argmax(losses_domain, axis=1)


def predict_once(filename):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get loss.
    # Create placeholder.
    
    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))
    domain_labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))

    loss_label, loss_domain = cifar10.inference(images) 

    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
  
    x_tr, y_tr, d_tr, nl_x_v, nl_y_v, nl_d_v, l_x_te, l_y_te, l_d_te = input.input()

    # Calculate predictions of training, non-lifelog validation, and lifelog test.
    tr_y_p, tr_d_p = predict_a_dataset(saver, filename, x_tr, y_tr, d_tr, images, labels, domain_labels, loss_label, loss_domain)
    nl_y_p, nl_d_p = predict_a_dataset(saver, filename, nl_x_v, nl_y_v, nl_d_v, images, labels, domain_labels, loss_label, loss_domain)
    l_y_p, l_d_p = predict_a_dataset(saver, filename, l_x_te, l_y_te, l_d_te, images, labels, domain_labels, loss_label, loss_domain)

    # Get lengths of results.
    L1 = len(tr_y_p)
    L2 = len(nl_y_p)
    L3 = len(l_y_p)

    y_tr=y_tr[:L1]
    d_tr=d_tr[:L1]
    nl_y_v=nl_y_v[:L2]
    nl_d_v=nl_d_v[:L2]
    l_y_te=l_y_te[:L3]
    l_d_te=l_d_te[:L3]

    return tr_y_p, tr_d_p, nl_y_p, nl_d_p, l_y_p, l_d_p, y_tr, d_tr, nl_y_v, nl_d_v, l_y_te, l_d_te

    #sp.savez('test.npz', tr_y_p=tr_y_p, tr_d_p=tr_d_p, nl_y_p=nl_y_p, nl_d_p=nl_d_p, l_y_p=l_y_p, l_d_p=l_d_p, \
    #        y_tr=y_tr[:L1], d_tr=d_tr[:L1], nl_y_v=nl_y_v[:L2], nl_d_v=nl_d_v[:L2], l_y_te=l_y_te[:L3], l_d_te=l_d_te[:L3])




def eval_once(filename):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get loss.
    # Create placeholder.
    
    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))
    domain_labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))

    #loss_label, loss_domain = cifar10.inference(images) 

    
    logits1, logits2= cifar10.inference(images)
    loss_label = cifar10.loss_without_decay(logits1, labels)
    loss_domain = cifar10.loss_without_decay(logits2, domain_labels)
    

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
  
    x_tr, y_tr, d_tr, nl_x_v, nl_y_v, nl_d_v, l_x_te, l_y_te, l_d_te = input.input()

    # Calculate losses of training, non-lifelog validation, and lifelog test.
    y_tr_loss, d_tr_loss = eval_a_dataset(saver, filename, x_tr, y_tr, d_tr, images, labels, domain_labels, loss_label, loss_domain)
    y_nl_loss, d_nl_loss = eval_a_dataset(saver, filename, nl_x_v, nl_y_v, nl_d_v, images, labels, domain_labels, loss_label, loss_domain)
    y_l_loss, d_l_loss = eval_a_dataset(saver, filename, l_x_te, l_y_te, l_d_te, images, labels, domain_labels, loss_label, loss_domain)


  return y_tr_loss, d_tr_loss, y_nl_loss, d_nl_loss, y_l_loss, d_l_loss




def test_once(filename):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get loss.
    # Create placeholder.
    
    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))

    loss_label, loss_domain = cifar10.inference(images)
    

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
   
    _, _, _, _, _, _, \
            x, y, domain= input.input()

    # Supplement the number of examples to multiplier of 25.
    num_of_examples = np.shape(x)[0]

    remainder = FLAGS.batch_size - int(math.ceil(num_of_examples/FLAGS.batch_size))
    index = range(num_of_examples) + [0] * remainder

    with tf.Session() as sess:
       #need to modify for test
       saver.restore(sess, filename) 
       global_step = int(filename.split('-')[1])


       # Allocate results in a list.
       losses_label = []
       losses_domain = []

       # Start the queue runners.
       step = 0
       while step + FLAGS.batch_size <= len(index):
           
           label_loss_value, domain_loss_value = sess.run([loss_label, loss_domain], feed_dict = {images:x[index[step:step+FLAGS.batch_size], :]})

           losses_label.append(label_loss_value)
           losses_domain.append(domain_loss_value)
           step = step + FLAGS.batch_size


       # Convert list of lists to numpy array.
       losses_label = np.asarray(losses_label)
       losses_domain = np.asarray(losses_domain)

       losses_label = losses_label.reshape((-1, 21))
       losses_domain = losses_domain.reshape((-1, 2))

       losses_label = losses_label[:num_of_examples, :]
       losses_domain = losses_domain[:num_of_examples, :]

  sp.savez('test.npz', losses_label = losses_label, losses_domain = losses_domain, y = y, domain = domain)

  return losses_label, losses_domain, y, domain


def evaluate():
  # select model files.
  step = np.asarray(range(61))
  step = step * 1000
  step[-1] = step[-1] - 1

  for i in step:
      print('step ', i, ' : ', eval_once('cifar10_train/model.ckpt-' + str(i)))


def test():
  print('in test')
  test_once('cifar10_train/model.ckpt-59999')


def predict():
    step = np.asarray(range(61))
    step = step * 1000
    step[-1] = step[-1] - 1

    result = []
    for i in step:
        result.append(predict_once('cifar10_train/model.ckpt-' + str(i)))

    np.savez('dp-update-predict.npz', result = result)


def main(argv=None):  # pylint: disable=unused-argument
  #evaluate()
  #predict_once('cifar10_train/model.ckpt-59999')
  predict()

if __name__ == '__main__':
  tf.app.run()
