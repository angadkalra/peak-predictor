from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import scipy.io as sio
import sklearn as skl
import tensorflow as tf
import numpy as np


def deepnn(x):
    # Reshape to use within a convolutional neural net.
    x_peaks = tf.reshape(x, [-1, 4, 251, 1])

    # First convolutional layer
    W_conv1 = weight_variable([4, 19, 1, 300])
    b_conv1 = bias_variable([300])
    conv_layer = conv2d(x_peaks, W_conv1)

    z1 = batch_normalization(conv_layer, 300,  1e-3)

    h_conv1 = tf.nn.relu(z1 + b_conv1)

    # Pooling layer
    h_pool1 = max_pool(h_conv1, 2, 3)

    # Second convolutional layer
    W_conv2 = weight_variable([2, 11, 300, 200])
    b_conv2 = bias_variable([200])
    conv_layer = conv2d(h_pool1, W_conv2)

    z2 = batch_normalization(conv_layer, 200, 1e-3)

    h_conv2 = tf.nn.relu(z2 + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool(h_conv2, 2, 4)

    # Third convolutional layer
    W_conv3 = weight_variable([1, 7, 200, 200])
    b_conv3 = bias_variable([200])
    conv_layer = conv2d(h_pool2, W_conv3)

    z3 = batch_normalization(conv_layer, 200, 1e-3)

    h_conv3 = tf.nn.relu(z3 + b_conv3)

    # Third pooling layer
    h_pool3 = max_pool(h_conv3, 1, 4)

    # Fully connected layer 1
    W_fc1 = weight_variable([6*200, 1000])
    b_fc1 = bias_variable([1000])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 6*200])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout1 - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully connected layer 2
    W_fc2 = weight_variable([1000, 1000])
    b_fc2 = bias_variable([1000])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Dropout2 - controls the complexity of the model, prevents co-adaptation of
    # features.
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Map the remaining features to 1 class
    W_fc3 = weight_variable([1000, 1])
    b_fc3 = bias_variable([1])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, dim1, dim2):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, dim1, dim2, 1],
                          strides=[1, dim1, dim2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization(x, dim, eps):
    """normalizes and returns batch"""
    mean, variance = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    scale = tf.Variable(tf.ones([dim]))
    offset = tf.Variable(tf.zeros([dim]))

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
    y_batch_pos = y_pos[pos_idxs, :]

    x_batch_neg = x_neg[neg_idxs, :]
    y_batch_neg = y_neg[neg_idxs, :]

    x_batch = np.concatenate((x_batch_pos, x_batch_neg))
    y_batch = np.concatenate((y_batch_pos, y_batch_neg))

    x_batch, y_batch = skl.utils.shuffle(x_batch,y_batch)

    return x_batch, y_batch


def import_training_data():
    """loads training and validation files and extracts training and validation sets"""
    train_pos = sio.loadmat('peaksBinTrainPos.mat')
    train_neg = sio.loadmat('peaksBinTrainNeg.mat')

    x_train_pos = train_pos['seq']
    y_train_pos = train_pos['labels']

    x_train_neg = train_neg['seq']
    y_train_neg = train_neg['labels']

    peaksBinValid = sio.loadmat('peaksBinValid.mat')
    x_valid = peaksBinValid['seq']
    y_valid = peaksBinValid['labels']

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

    batch_size = 50

    # Create the model
    x = tf.placeholder(tf.float32, [None, 1004])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)

    y_hat = tf.greater(y_conv, 0.5)
    correct_prediction = tf.equal(y_hat, tf.equal(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    precision = tf.metrics.precision(y_, y_hat)
    recall = tf.metrics.recall(y_, y_hat)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(100):
            batch = next_training_batch(train_data,  batch_size)
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.3})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.3})

        # print('test accuracy %g' % accuracy.eval(feed_dict={x: train_data['x_valid'], y_: train_data['y_valid'], keep_prob: 0.3}))

        saver.save(sess, "models/model")

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
