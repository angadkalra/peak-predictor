from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import scipy.io
import sklearn as skl
import tensorflow as tf
import numpy as np

FLAGS = None


def deepnn(x):
    # Reshape to use within a convolutional neural net.
    x_peaks = tf.reshape(x, [-1, 4, 251, 1])

    # First convolutional layer
    W_conv1 = weight_variable([2, 2, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_peaks, W_conv1) + b_conv1)

    # Pooling layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1
    W_fc1 = weight_variable([63*64, 1004])
    b_fc1 = bias_variable([1004])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 63*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1004 features to 1 class
    W_fc2 = weight_variable([1004, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def next_training_batch(X,y,size):
    """next_training_batch generates a batch of random training examples."""
    x_batch = X[np.random.choice(X.shape[0], size, False), :]
    x_batch = np.asarray(x_batch.todense())
    x_batch = x_batch.astype(int)

    y_batch = y[np.random.choice(y.shape[0], size, False)]

    return (x_batch, y_batch)

def main(_):
    # Import data
    peaksBin = scipy.io.loadmat('peaksBin.mat')
    X = peaksBin['seq']
    y = peaksBin['labels']
    X, y = skl.utils.shuffle(X, y)

    X_train = X[1:43138, :]
    y_train = y[1:43138]

    X_valid = X[43138:53138, :]
    y_valid = y[43138:53138]

    X_test = X[53138:, :]
    y_test = y[53138:]

    # Create the model
    x = tf.placeholder(tf.float32, [None, 1004])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch = next_training_batch(X_train, y_train, 50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
