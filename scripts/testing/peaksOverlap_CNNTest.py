import sys

import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def main(_):

    sess = tf.Session()

    model = tf.train.import_meta_graph('../../models/tempModel.meta')
    model.restore(sess, tf.train.latest_checkpoint('../../models'))

    graph = tf.get_default_graph()

    # Define placeholders
    x = graph.get_operation_by_name('input')
    x = x.values()[0]

    y_ = graph.get_operation_by_name('labels')
    y_ = y_.values()[0]

    keep_prob = graph.get_operation_by_name('keep_prob')
    keep_prob = keep_prob.values()[0]

    # Load data
    peaksBinTest = sio.loadmat('../../data/testing/peaksBinTest.mat')

    test_seq = peaksBinTest['seq']
    test_seq = np.asarray(test_seq.todense()).astype(np.float32)
    test_labels = np.loadtxt('../../data/testing/olapLabelsTest')

    # Define prediction error and plot results
    y_conv = graph.get_operation_by_name('y_conv')
    y_conv = y_conv.values()[0]

    y_hat = y_conv.eval(session=sess, feed_dict={x: test_seq[:100, :], keep_prob: 0.3})

    # plot y_hat and y_valid
    plt.plot(y_hat, 'r', test_labels[:100], 'b')
    plt.show()

    error = graph.get_operation_by_name('l2_loss')
    error = error.values()[0]

    test_error = error.eval(session=sess, feed_dict={x: test_seq[:100], y_: test_labels[:100], keep_prob: 0.3})

    print('test error %g' % test_error)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])