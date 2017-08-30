import sys

import tensorflow as tf
import scipy.io as sio
import numpy as np


def main(_):

    sess = tf.Session()

    model = tf.train.import_meta_graph('../../models/olapModel4.meta')
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
    test_labels = peaksBinTest['labels']

    # Define accuracy prediction
    y_conv = graph.get_operation_by_name('y_conv').values()[0]
    y_hat = tf.greater(y_conv, 0.5)

    correct_prediction = tf.equal(y_hat, tf.equal(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # preds = tf.cast(y_hat, tf.float32)
    # true_pos = tf.count_nonzero(y_ * preds)
    # false_pos = tf.count_nonzero((y_ - 1) * preds)
    # false_neg = tf.count_nonzero(y_ * (preds - 1))

    # tp = true_pos.eval(session=sess, feed_dict={x: test_seq[:100, :], y_: test_labels[:100], keep_prob:0.3})
    # fp = false_pos.eval(session=sess, feed_dict={x: test_seq[:100, :], y_: test_labels[:100], keep_prob:0.3})
    # fn = false_neg.eval(session=sess, feed_dict={x: test_seq[:100, :], y_: test_labels[:100], keep_prob:0.3})

    # Depending on which model you restore, y_ is either a tensor with 2 dim (i.e [?, 1]) or 1 dim (i.e [?,]). If 2 dim,
    # then change test_labels[:100, 0] to test_labels[:100]
    test_accuracy = accuracy.eval(session=sess, feed_dict={x: test_seq[:100, :], y_: test_labels[:100, 0], keep_prob:0.3})

    # print('test accuracy %g, tp %g, fp %g, fn %g' % (test_accuracy, tp, fp, fn))
    print('test accuracy %g' % test_accuracy)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
