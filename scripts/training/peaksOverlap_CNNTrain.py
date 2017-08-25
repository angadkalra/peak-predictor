from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import scipy.io as sio
import sklearn as skl
import tensorflow as tf
import numpy as np
import time
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def deepnn(x):
    # Reshape to use within a convolutional neural net.
    x_peaks = tf.reshape(x, [-1, 4, 251, 1])

    # First convolutional layer
    W_conv1 = weight_variable([4, 19, 1, 300], 'W_conv1')
    b_conv1 = bias_variable([300], 'b_conv1')
    conv_layer = conv2d(x_peaks, W_conv1)

    scale1 = tf.Variable(tf.ones([300]), name='scale1')
    offset1 = tf.Variable(tf.zeros([300]), name='offset1')
    z1 = batch_normalization(conv_layer + b_conv1, scale1, offset1, 1e-3)

    h_conv1 = tf.nn.relu(z1, name='h_conv1')

    # Pooling layer
    h_pool1 = max_pool(h_conv1, 2, 3, 'h_pool1')

    # Second convolutional layer
    W_conv2 = weight_variable([2, 11, 300, 200], 'W_conv2')
    b_conv2 = bias_variable([200], 'b_conv2')
    conv_layer = conv2d(h_pool1, W_conv2)

    scale2 = tf.Variable(tf.ones([200]), name='scale2')
    offset2 = tf.Variable(tf.zeros([200]), name='offset2')
    z2 = batch_normalization(conv_layer + b_conv2, scale2, offset2, 1e-3)

    h_conv2 = tf.nn.relu(z2, 'h_conv2')

    # Second pooling layer.
    h_pool2 = max_pool(h_conv2, 2, 4, 'h_pool2')

    # Third convolutional layer
    W_conv3 = weight_variable([1, 7, 200, 200], 'W_conv3')
    b_conv3 = bias_variable([200], 'b_conv3')
    conv_layer = conv2d(h_pool2, W_conv3)

    scale3 = tf.Variable(tf.ones([200]), name='scale3')
    offset3 = tf.Variable(tf.zeros([200]), name='offset3')
    z3 = batch_normalization(conv_layer + b_conv3, scale3, offset3, 1e-3)

    h_conv3 = tf.nn.relu(z3, 'h_conv3')

    # Third pooling layer
    h_pool3 = max_pool(h_conv3, 1, 4, 'h_pool3')

    # Fully connected layer 1
    W_fc1 = weight_variable([6*200, 1000], 'W_fc1')
    b_fc1 = bias_variable([1000], 'b_fc1')

    h_pool3_flat = tf.reshape(h_pool3, [-1, 6*200])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name='h_fc1')

    # Dropout1 - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    # Fully connected layer 2
    W_fc2 = weight_variable([1000, 1000], 'W_fc2')
    b_fc2 = bias_variable([1000], 'b_fc2')

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='h_fc2')

    # Dropout2 - controls the complexity of the model, prevents co-adaptation of
    # features.
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name='h_fc2_drop')

    # Map the remaining features to 1 class
    W_fc3 = weight_variable([1000, 1], 'W_fc3')
    b_fc3 = bias_variable([1], 'b_fc3')

    output = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name="output")
    y_conv = tf.sigmoid(output, name='y_conv')

    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, dim1, dim2, var_name):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, dim1, dim2, 1],
                          strides=[1, dim1, dim2, 1], padding='SAME', name=var_name)


def weight_variable(shape, var_name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=var_name)


def batch_normalization(x, scale, offset, eps):
    """normalizes and returns batch"""
    mean, variance = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    bn = tf.nn.batch_normalization(x, mean, variance, scale, offset, eps)
    return bn


def next_training_batch(data, size):
    """next_training_batch generates a batch of random training examples."""
    x_pos = data['x_pos']
    y_pos = data['y_pos']
    x_neg = data['x_neg']
    y_neg = data['y_neg']

    pos_idxs = np.random.choice(x_pos.shape[0], int(size/2), False)
    neg_idxs = np.random.choice(x_neg.shape[0], int(size/2), False)

    x_batch_pos = x_pos[pos_idxs, :]
    y_batch_pos = y_pos[pos_idxs]

    x_batch_neg = x_neg[neg_idxs, :]
    y_batch_neg = y_neg[neg_idxs]

    x_batch = np.concatenate((x_batch_pos, x_batch_neg))
    y_batch = np.concatenate((y_batch_pos, y_batch_neg))

    x_batch, y_batch = skl.utils.shuffle(x_batch,y_batch)

    return x_batch, y_batch


def import_training_data():
    """loads training and validation files and extracts training and validation sets"""

    # Training
    train_data = sio.loadmat('../../data/training/peaksBinTrain.mat')
    train_labels = np.loadtxt('../../data/training/olapLabelsTrain')

    train_data_seq = train_data['seq']

    x_train_pos = train_data_seq[train_labels > 0, :]
    y_train_pos = train_labels[train_labels > 0]

    x_train_neg = train_data_seq[train_labels == 0, :]
    y_train_neg = train_labels[train_labels == 0]

    # Validation
    valid_data = sio.loadmat('../../data/training/peaksBinTrain.mat')

    x_valid = valid_data['seq']
    y_valid = np.loadtxt('../../data/training/olapLabelsValid')

    # Want dense numpy ndarray
    x_train_pos = np.asarray(x_train_pos.todense()).astype(int)
    x_train_neg = np.asarray(x_train_neg.todense()).astype(int)
    x_valid = np.asarray(x_valid.todense()).astype(int)

    train_data = {}
    train_data['x_pos'] = x_train_pos
    train_data['y_pos'] = y_train_pos
    train_data['x_neg'] = x_train_neg
    train_data['y_neg'] = y_train_neg
    train_data['x_valid'] = x_valid
    train_data['y_valid'] = y_valid

    return train_data


def main(_):
    # Import training and validation data
    train_data = import_training_data()

    x_valid = train_data['x_valid']
    y_valid = train_data['y_valid']

    batch_size = 50

    # Create the model
    x = tf.placeholder(tf.float32, [None, 1004], name='input')
    y_ = tf.placeholder(tf.float32, [None, ], name='labels')

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    # Define performance stats and optimizer
    error = tf.reduce_mean(tf.nn.l2_loss(y_ - y_conv, name='l2_loss'))
    train_step = tf.train.RMSPropOptimizer(1e-3).minimize(error)

    # Save model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        start = time.time()

        for i in range(100):
            batch = next_training_batch(train_data,  batch_size)
            if i % 10 == 0:
                train_error = error.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.3})

                print('step %d, training error %g' % (i, train_error))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.3})

        end = time.time()
        print('Training time %g seconds' % (end - start))

        saver.save(sess, "../models/tempModel")

        test_error = error.eval(feed_dict={x: x_valid[:100, :], y_: y_valid[:100], keep_prob: 0.3})
        print('test error %g' % test_error)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
