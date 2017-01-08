# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
import scipy as sp
import scipy.io

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import input

IMAGE_SIZE = 112

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 60000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train',
        """Directory where to read model checkpoints.""")


def tower_loss_label(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits, _ = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])


  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def tower_loss_domain(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  _, logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])


  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is not None:
         # Add 0 dimension to the gradients to represent the tower.
         expanded_g = tf.expand_dims(g, 0)

         # Append on a 'tower' dimension which we will average over below.
         grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_sum(grad, 0)

    #print('grads: ', grads[0].name)
    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    '''
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    '''

    # Tradeoff is the lambda from the paper.
    tradeoff = tf.placeholder(tf.float32, shape=[])

    # Create an optimizer that performs gradient descent.
    lr = tf.placeholder(tf.float32, shape=[])
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))
    domain_labels = tf.placeholder(tf.int32, shape = (FLAGS.batch_size,))


    # Calculate the gradients for each model tower.
    for i in [0,]:
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss_label = tower_loss_label(scope, images, labels)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads_label = opt.compute_gradients(loss_label)


    for i in [1,]:
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          tf.get_variable_scope().reuse_variables()
          loss_domain = tower_loss_domain(scope, images, domain_labels)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads_domain = opt.compute_gradients(loss_domain)


    # Apply the gradients to adjust the shared variables.
    #test = tf.get_default_graph().get_tensor_by_name("ExpandDims_8:0")
    #capped_grads1 = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_label if grad is not None]
    #apply_gradient_op1 = opt.apply_gradients(capped_grads1, global_step=global_step)

    # change gradients of domain classifier.
    capped_grads2 = []
    for grad, var in grads_domain:
        if any(s in var.name for s in ['conv1', 'conv2']):
            grad = tf.mul(grad, tradeoff)
            capped_grads2.append((grad, var))

        else:
            capped_grads2.append((grad, var))

    grads = average_gradients([grads_label, capped_grads2])
    #grads = grads_label
    capped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads if grad is not None]
    apply_gradient_op = opt.apply_gradients(capped_grads, global_step=global_step)


    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)
    #train_op2 = tf.group(apply_gradient_op2, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep = None)


    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)


    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #_, loss_value = sess.run([train_op, loss])
      x_train, y_train, domain_train = input.distorted_input()
      

      # Feed images, labels, domain_labels; learning_rate and tradeoff.
      to_value = 1 - 2/(1.0 + np.exp(-10.0 * step/60000))
      lr_value = 0.01/np.power((1 + 10.0 * step/60000), 0.75)

      _, loss_value = sess.run([train_op, loss_label], \
              feed_dict={images:x_train, labels:y_train, domain_labels:domain_train,\
              tradeoff:to_value, lr:lr_value})
      
 
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch), lr=%.3f, to=%.3f')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, lr_value, to_value))
      
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
