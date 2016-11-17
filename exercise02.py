import tensorflow as tf
import numpy as np
# Loads mnist data set.
# Each image is a row and multiple rows make up one batch.
# The class with label i is a 0-vector except at index i, where it is 1.
# Learning labels with regression instead of classification would be harder.
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

image_size = 28*28
n_labels = 10
hidden_size = 500
# Larger batch size => higher accuracy, less overfitting, longer waiting
batch_size = 1000
mnist = read_data_sets("data", one_hot=True)

def f(X, n_in, n_out):
    # f(X) = X.dot(W) where W is of shape (n_in, n_out)
    W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0/np.sqrt(n_in)))
    b = tf.Variable(tf.truncated_normal([n_out], stddev=1.0/np.sqrt(n_out)))
    y = tf.matmul(X, W) + b
    return y

# Define neural network graph.
X = tf.placeholder(tf.float32, [None, image_size])
y = tf.nn.relu(f(X, image_size, hidden_size))
y = tf.nn.relu(f(y, hidden_size, hidden_size))
# tanh here produces slightly better accuracy than relu everywhere.
y = tf.nn.tanh(f(y, hidden_size, hidden_size))
y = f(y, hidden_size, n_labels)

labels = tf.placeholder(tf.float32, [None, n_labels])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels))
train = tf.train.AdamOptimizer().minimize(loss)
sess = tf.InteractiveSession()

tf.initialize_all_variables().run()
for n_batches in range(1000+1): # Run for 1001 epochs.
    batch_X, batch_labels = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={X: batch_X, labels: batch_labels})
    if n_batches % 100 == 0: # Print accuracy sometimes.
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        feed_dict = {X: mnist.test.images, labels: mnist.test.labels}
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print("%f accuracy after batch %5d"%(acc, n_batches))
