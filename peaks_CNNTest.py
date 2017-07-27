import sys

import tensorflow as tf
import scipy.io as sio
import numpy as np


def deepnn(x, session):

    graph = tf.get_default_graph()

    # Reshape to use within a convolutional neural net.
    x_peaks = tf.reshape(x, [-1, 4, 251, 1])

    # First convolutional layer
    W_conv1 = graph.get_tensor_by_name('Variable:0')
    b_conv1 = graph.get_tensor_by_name('Variable_1:0')
    conv_layer = conv2d(x_peaks, W_conv1)

    scale1 = graph.get_tensor_by_name('Variable_2:0')
    offset1 = graph.get_tensor_by_name('Variable_3:0')
    z1 = batch_normalization(conv_layer, scale1, offset1, 300,  1e-3)

    h_conv1 = tf.nn.relu(z1 + b_conv1, name='h_conv1')

    # Pooling layer
    h_pool1 = max_pool(h_conv1, 2, 3, 'h_pool1')

    # Second convolutional layer
    W_conv2 = graph.get_tensor_by_name('Variable_4:0')
    b_conv2 = graph.get_tensor_by_name('Variable_5:0')
    conv_layer = conv2d(h_pool1, W_conv2)

    scale2 = graph.get_tensor_by_name('Variable_6:0')
    offset2 = graph.get_tensor_by_name('Variable_7:0')
    z2 = batch_normalization(conv_layer, scale2, offset2, 200, 1e-3)

    h_conv2 = tf.nn.relu(z2 + b_conv2, 'h_conv2')

    # Second pooling layer.
    h_pool2 = max_pool(h_conv2, 2, 4, 'h_pool2')

    # Third convolutional layer
    W_conv3 = graph.get_tensor_by_name('Variable_8:0')
    b_conv3 = graph.get_tensor_by_name('Variable_9:0')
    conv_layer = conv2d(h_pool2, W_conv3)

    scale3 = graph.get_tensor_by_name('Variable_10:0')
    offset3 = graph.get_tensor_by_name('Variable_11:0')
    z3 = batch_normalization(conv_layer, scale3, offset3, 200, 1e-3)

    h_conv3 = tf.nn.relu(z3 + b_conv3, 'h_conv3')

    # Third pooling layer
    h_pool3 = max_pool(h_conv3, 1, 4, 'h_pool3')

    # Fully connected layer 1
    W_fc1 = graph.get_tensor_by_name('Variable_12:0')
    b_fc1 = graph.get_tensor_by_name('Variable_13:0')

    h_pool3_flat = tf.reshape(h_pool3, [-1, 6*200])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name='h_fc1')

    # Dropout1 - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = 0.3
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    # Fully connected layer 2
    W_fc2 = graph.get_tensor_by_name('Variable_14:0')
    b_fc2 = graph.get_tensor_by_name('Variable_15:0')

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='h_fc2')

    # Dropout2 - controls the complexity of the model, prevents co-adaptation of
    # features.
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name='h_fc2_drop')

    # Map the remaining features to 1 class
    W_fc3 = graph.get_tensor_by_name('Variable_16:0')
    b_fc3 = graph.get_tensor_by_name('Variable_17:0')

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    return y_conv


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


def batch_normalization(x, scale, offset, dim, eps):
    """normalizes and returns batch"""
    mean, variance = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    bn = tf.nn.batch_normalization(x, mean, variance, scale, offset, eps)
    return bn


def main(_):

    sess = tf.Session()

    model = tf.train.import_meta_graph('models/model.meta')
    model.restore(sess, tf.train.latest_checkpoint('models'))

    graph = tf.get_default_graph()

    # Define placeholders
    x = graph.get_operation_by_name('Placeholder')
    x = x.values()[0]

    y_ = graph.get_operation_by_name('Placeholder_1')
    y_ = y_.values()[0]

    keep_prob = graph.get_operation_by_name('Placeholder_2')
    keep_prob = keep_prob.values()[0]

    # Load data
    peaksBinTest = sio.loadmat('peaksBinTest.mat')

    test_seq = peaksBinTest['seq']
    test_seq = np.asarray(test_seq.todense()).astype(np.float32)
    test_labels = peaksBinTest['labels']

    # Define accuracy prediction
    y_conv = deepnn(test_seq[1000:2000, :], session=sess)

    y_hat = tf.greater(y_conv, 0.5)
    correct_prediction = tf.equal(y_hat, tf.equal(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_accuracy = accuracy.eval(session=sess, feed_dict={x: test_seq[1000:2000, :], y_: test_labels[1000:2000], keep_prob:0.3})

    print('test accuracy % g' % test_accuracy)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])