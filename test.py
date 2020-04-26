import tensorflow as tf
import numpy as np

x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)

    s1 = (indices, values, shape)
    s2 = (indices, values, shape)
    s3 = np.vstack((s1, s2))
    print(sess.run(y, feed_dict={
        x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.
    print(sess.run(y, feed_dict={
        x: (indices, values, shape)}))  # Will succeed.
    print(sess.run(y, feed_dict={
        x: s3}))  # Will succeed.