#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# all this work for just 98.7% accuracy

image_size = 28*28
n_labels = 10
batch_size = 1000

mnist = read_data_sets("../mnist", one_hot=True)

def variable_summaries(var, name):
    with tf.name_scope(name.replace(' ', '_')):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def make_weights(shape, name='unnamed_variable'):
    variable = tf.Variable(tf.truncated_normal(shape, stddev=1.0/np.sqrt(shape[0])))
    variable_summaries(variable, name)
    return variable

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def display_kernels(kernels, name):
    name = name.replace(' ', '_')
    with tf.variable_scope(name):
        kernels_min = tf.reduce_min(kernels)
        kernels_max = tf.reduce_max(kernels)
        kernels = (kernels - kernels_min)/(kernels_max - kernels_min)
        kernels = tf.transpose(kernels, [3, 0, 1, 2])
        tf.image_summary('kernels', kernels, max_images=3)

# so many weights
W_conv1 = make_weights([5, 5, 1, 8], 'first conv layer weights')
b_conv1 = make_weights([8], 'first conv layer biases')
W_conv2 = make_weights([5, 5, 8, 9], 'second conv layer weights')
b_conv2 = make_weights([9], 'second conv layer biases')
flat_size = 7 * 7 * 9
W_fc1 = make_weights([flat_size, 500], 'first fully connected layer weights')
b_fc1 = make_weights([500], 'first fully connected layer biases')
W_fc2 = make_weights([500, 100], 'second fully connected layer weights')
b_fc2 = make_weights([100], 'second fully connected layer biases')
W_fc3 = make_weights([100, 10], 'third fully connected layer weights')
b_fc3 = make_weights([10], 'third fully connected layer biases')

display_kernels(W_conv1, 'first conv layer kernels')

# placeholders
x = tf.placeholder(tf.float32, [None, image_size])
y = tf.placeholder(tf.float32, [None, n_labels])
# this is for dropout, but dropout does not help
keep_prob = tf.placeholder(tf.float32)

# we get vector data, but conv layer needs image-shaped data
x_image = tf.reshape(x, [-1, 28, 28, 1])

# pool(relu(conv(X, W) + b))
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
h_pool1 = max_pool_2x2(h_conv1)
# another one
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_conv2 = tf.nn.dropout(h_conv2, keep_prob)
h_pool2 = max_pool_2x2(h_conv2)

# flatten the images back to vectors
h_fc0 = tf.reshape(h_pool2, [-1, flat_size])
# relu(XW + b)
h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
#h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
# relu(XW + b)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
#h_fc2 = tf.nn.dropout(h_fc2, keep_prob)
# XW + b
y_conv =           tf.matmul(h_fc2, W_fc3) + b_fc3

# softmax cross entropy loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))
tf.summary.scalar('loss', loss)

# accuracy testing network
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# adam optimizer is best optimizer (maybe)
train = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()

summaries = tf.merge_all_summaries()
summaries_directory = "summaries"
train_writer = tf.train.SummaryWriter(summaries_directory + "/train", sess.graph)
test_writer = tf.train.SummaryWriter(summaries_directory + "/test", sess.graph)

# tensorflow magic
tf.global_variables_initializer().run()

for batch in range(1000+1):
    print("batch %d"%batch)
    # train
    batch_train_x, batch_train_y = mnist.train.next_batch(batch_size)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_ = sess.run(
        [summaries, train],
        options=run_options,
        run_metadata=run_metadata,
        feed_dict={x: batch_train_x, y: batch_train_y, keep_prob: 0.75})
    train_writer.add_summary(summary, batch)
    train_writer.add_run_metadata(run_metadata, "batch %d"%batch)
    
    if batch % 100 == 0:
        # calculate accuracy
        feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}
        summary, acc = sess.run([summaries, accuracy], feed_dict=feed_dict)
        test_writer.add_summary(summary, batch)
        print("%f accuracy"%acc)
        
