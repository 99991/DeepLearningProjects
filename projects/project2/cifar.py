import cPickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

image_width = 32
image_height = 32
num_channels = 3
num_labels = 10
learning_rate = 0.001
batch_size = 100
kernel1_size = 3
num_kernels1 = 5
num_hidden = 100
seed = 0
num_batches = 100

np.random.seed(0)

# TODO
# load all batches
# normalize data
# generate random data
# pool
# dropout
# different architectures
# test error
# TensorBoard :(

def unpickle(path):
    with open(path, "rb") as f:
        return cPickle.load(f)

def hot_encode(labels):
    n = len(labels)
    encoded_labels = np.zeros((n, num_labels), dtype=np.int32)
    encoded_labels[np.arange(n), labels] = 1
    return encoded_labels

directory = "cifar-10-batches-py"

label_names = unpickle(directory + "/batches.meta")['label_names']
cifar = unpickle(directory + "/data_batch_1")
cifar_rows = np.array(cifar['data'], dtype=np.float32)/256.0
cifar_labels = hot_encode(np.array(cifar['labels']))

print(cifar_rows.shape)

def row_to_image(row):
    image = np.zeros((image_width, image_height, num_channels), dtype=row.dtype)
    n = image_width*image_height
    # there must be a better way to convert RRR GGG BBB to RGB RGB RGB
    image[:, :, 0] = row[0*n:1*n].reshape((image_width, image_height))
    image[:, :, 1] = row[1*n:2*n].reshape((image_width, image_height))
    image[:, :, 2] = row[2*n:3*n].reshape((image_width, image_height))
    return image

def draw_grid(rows, labels, nx, ny, offset=0):
    for i in range(nx*ny):
        row = rows[i + offset]
        label = labels[i + offset]
        
        plt.subplot(ny, nx, i + 1)
        plt.title(label_names[label])
        plt.axis('off')
        plt.imshow(row_to_image(row))
    plt.show()

def next_batch(n):
    indices = np.random.randint(len(cifar_rows), size=n)
    rows = cifar_rows[indices, :]
    labels = cifar_labels[indices]
    return rows, labels

# placeholders for training and testing data
rows       = tf.placeholder(tf.float32, [None, image_width*image_height*num_channels])
labels     = tf.placeholder(tf.float32, [None, num_labels])
#keep_prob = tf.placeholder(tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate)

flat_size = image_width*image_height*num_kernels1

conv1_weights = tf.Variable(tf.truncated_normal([kernel1_size, kernel1_size, 3, num_kernels1], stddev=0.1, seed=seed))
conv1_biases  = tf.Variable(tf.constant(0.0, tf.float32, [num_kernels1]))
#conv2_weights = tf.Variable(tf.truncated_normal([kernel2_size, kernel2_size, num_kernels1, num_kernels2], stddev=0.1, seed=seed))
#conv2_biases  = tf.Variable(tf.constant(0.1, tf.float32, [num_kernels2]))
fc1_weights   = tf.Variable(tf.truncated_normal([flat_size, num_hidden], stddev=0.1, seed=seed))
fc1_biases    = tf.Variable(tf.constant(0.1, tf.float32, [num_hidden]))
fc2_weights   = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1, seed=seed))
fc2_biases    = tf.Variable(tf.constant(0.1, tf.float32, [num_labels]))

X = rows
# reshape data to image shape
#X = tf.nn.dropout(X, keep_prob)
X = tf.reshape(X, [-1, image_width, image_height, 3])
# conv relu pool
X = tf.nn.conv2d(X, conv1_weights, strides=[1,1,1,1], padding='SAME')
X = tf.nn.relu(tf.nn.bias_add(X, conv1_biases))
#X = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# conv relu pool
#X = tf.nn.conv2d(X, conv2_weights, strides=[1,1,1,1], padding='SAME')
#X = tf.nn.relu(tf.nn.bias_add(X, conv2_biases))
#X = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# flatten data to row shape
X = tf.reshape(X, [-1, flat_size])
# relu(XW + b)
X = tf.nn.relu(tf.matmul(X, fc1_weights) + fc1_biases)
# dropout
#X = tf.nn.dropout(X, keep_prob, seed=seed)
# XW + b
X = tf.matmul(X, fc2_weights) + fc2_biases
Y = X

"""
# regularization to keep weights small
regularization = (
        tf.nn.l2_loss(fc1_weights) +
        tf.nn.l2_loss(fc2_weights) +
        tf.nn.l2_loss(fc1_biases ) +
        tf.nn.l2_loss(fc2_biases ))
"""
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, labels))
#loss += regularization_factor*regularization
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train = optimizer.minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


accuracies = []
losses = []
for batch in range(num_batches):
    batch_rows, batch_labels = next_batch(batch_size)
    feed_dict = {rows:batch_rows, labels:batch_labels}#, keep_prob:dropout_keep_probability}
    sess.run([train], feed_dict=feed_dict)
    acc = sess.run(accuracy, feed_dict=feed_dict)
    print(acc)
