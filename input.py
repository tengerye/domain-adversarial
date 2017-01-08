from six.moves import xrange
import numpy as np
import scipy as sp
import scipy.io

import tensorflow as tf

import random

FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 112
NUM_CLASSES = 21
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

file = sp.load('/ichec/work/dcu01/tenger/project/Lifelogs/C2/data/lifelog.npz')

# load all data from files.
lifelog_x_train = np.float32(file['x_train'])
lifelog_x_test = np.float32(file['x_test'])
lifelog_y_train = file['y_train']
lifelog_y_test = file['y_test']
lifelog_domain_train = lifelog_y_train * 0 + 1
lifelog_domain_test = lifelog_y_test * 0 + 1

file = sp.load('/ichec/work/dcu01/tenger/project/Lifelogs/C2/data/non-lifelog.npz')
non_lifelog_x_train = np.float32(file['x_train'])
non_lifelog_x_validate = np.float32(file['x_validate'])
non_lifelog_x_test = np.float32(file['x_test'])
non_lifelog_y_train = file['y_train']
non_lifelog_y_validate = file['y_validate']
non_lifelog_y_test = file['y_test']
non_lifelog_domain_train = non_lifelog_y_train * 0
non_lifelog_domain_validate = non_lifelog_y_validate * 0
non_lifelog_domain_test = non_lifelog_y_test * 0

# Put training data together.
x_train = np.concatenate([lifelog_x_train, non_lifelog_x_train])
y_train = np.concatenate([lifelog_y_train, non_lifelog_y_train])
domain_train = np.concatenate([lifelog_domain_train, non_lifelog_domain_train])

# Clean memory.
lifelog_x_train = None
non_lifelog_x_train = None
lifelog_y_train = None
non_lifelog_y_train = None
lifelog_domain_train = None
non_lifelog_domain_train = None

print('finish loading data...')

# 1. The input is labelled data from source and target data.
# The function randomly select several images from the input and do some pre-processing.
def distorted_input():
    indices = np.random.choice(x_train.shape[0], FLAGS.batch_size)
    images = x_train[indices, :, :, :]
    labels = y_train[indices]
    domain_labels = domain_train[indices]
   
    # randomly corupt images.
    for i in range(FLAGS.batch_size):
        images[i, :, :, :] = image_distort(images[i, :, :, :])

    return images, labels, domain_labels

# 3. Randomly choose batch size of examples for evaluation.
def batch_input():
    # From training set (mix).
    indices = np.random.choice(x_train.shape[0], FLAGS.batch_size)
    train_images = x_train[indices, :, :, :]
    train_labels = y_train[indices]
    train_domain = domain_train[indices]

    # From test set of lifelog.
    indices = np.random.choice(x_train.shape[0], FLAGS.batch_size)
    test_t_images = lifelog_x_test[indices, :, :, :]
    test_t_labels = lifelog_y_test[indices]
    test_t_domain = lifelog_domain_test[indices]

    # From validation set of non-lifelog.
    indices = np.random.choice(x_train.shape[0], FLAGS.batch_size)
    test_s_images = non_lifelog_x_validate[indices, :, :, :]
    test_s_labels = non_lifelog_y_validate[indices]
    test_s_domain = non_lifelog_domain_validate[indices]
   
    return train_images, train_labels, train_domain, test_t_images, test_t_labels, test_t_domain, test_s_images, test_s_labels, test_s_domain


# 2. The input facilitates validation and test, and provides following data(all of the data):
# training data, label, domain; source validate data, label, domain; target test data, label, domain, domain.
def input():
    # Perform normalization to all images.
    for i in range(np.shape(x_train)[0]):
        x_train[i, :, :, :] = normalize_image(x_train[i, :, :, :])

    for i in range(np.shape(non_lifelog_x_validate)[0]):                        
        non_lifelog_x_validate[i, :, :, :] = normalize_image(non_lifelog_x_validate[i, :, :, :])

    for i in range(np.shape(lifelog_x_test)[0]):
        lifelog_x_test[i, :, :, :] = normalize_image(lifelog_x_test[i, :, :, :])

    return x_train, y_train, domain_train, non_lifelog_x_validate, \
            non_lifelog_y_validate, non_lifelog_domain_validate, \
            lifelog_x_test, lifelog_y_test, lifelog_domain_test


# distort single image.
def image_distort(distorted_image):
    # randomly flip horizontally.
    if random.random() < 0.5:
        distorted_image = distorted_image[:, ::-1, :]

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = distorted_image + random.uniform(-63, 63)

    channel_mean = distorted_image.mean(axis=2, keepdims=True)
    distorted_image = (distorted_image - channel_mean) * random.uniform(0.2, 1.8) + channel_mean


    # Subtract off the mean and divide by the variance of the pixels.
    float_image = (distorted_image - distorted_image.mean())/max(np.std(distorted_image), 1.0/np.sqrt(112 * 112 * 3))


    return float_image

# Normalize an image to be according to those of training.
def normalize_image(image):
    return (image - image.mean())/max(np.std(image), 1.0/np.sqrt(112 * 112 * 3))
