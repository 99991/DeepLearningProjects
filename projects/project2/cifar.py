import time
import cPickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# variables
learning_rate     = 0.001
batch_size        = 200
num_batches       = 10000
num_kernels       = [8, 8, 16, 16, 32, 32, 32, 32]
kernel_sizes      = [3, 3, 3, 3, 3, 3, 3, 3]
hidden_sizes      = [500, 100]
dropout_keep_prob = 1.0
more_layers       = False

# constants
image_width = 32
image_height = 32
num_channels = 3
num_labels = 10
seed = 0 

np.random.seed(0)

cifar_directory = "../../cifar-10-batches-py"

# TODO
# normalize data    
# generate random data
# TensorBoard :(

def unpickle(path):
    with open(path, "rb") as f:
        return cPickle.load(f)

def hot_encode(labels):
    n = len(labels)
    encoded_labels = np.zeros((n, num_labels), dtype=np.int32)
    encoded_labels[np.arange(n), labels] = 1
    return encoded_labels

def load_batch(name):
    batch = unpickle(cifar_directory + "/" + name)
    images = np.array(batch['data'], dtype=np.float32)/256.0
    # cifar dataset is in RRR GGG BBB, RRR GGG BBB, ... format
    images = np.reshape(images, [10000, num_channels, image_width, image_height])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = hot_encode(np.array(batch['labels']))
    return images, labels

# load cifar dataset
train_images = []
train_labels = []
for i in range(1, 6):
    images, labels = load_batch("data_batch_%i"%i)
    train_images.append(images)
    train_labels.append(labels)
test_images, test_labels = load_batch("test_batch")
train_images = np.concatenate(train_images, 0)
train_labels = np.concatenate(train_labels, 0)

label_names = unpickle(cifar_directory + "/batches.meta")['label_names']

def draw_grid(images, labels, nx, ny, offset=0):
    # Example:
    # draw_grid(train_images, train_labels, 4, 4)
    for i in range(nx*ny):
        image = images[i + offset]
        label = labels[i + offset]
        
        plt.subplot(ny, nx, i + 1)
        plt.title(label_names[np.argmax(label)])
        plt.axis('off')
        plt.imshow(image)
    plt.show()

def next_batch(n):
    indices = np.random.randint(len(train_images), size=n)
    images = train_images[indices]
    labels = train_labels[indices]
    return images, labels

# placeholders for training and testing data
images    = tf.placeholder(tf.float32, [None, image_width, image_height, num_channels])
labels    = tf.placeholder(tf.float32, [None, num_labels])
keep_prob = tf.placeholder(tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate)

#flat_size = image_width*image_height*num_kernels1

def conv(X, kernel_size, num_kernels, prev_num_kernels):
    W = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, prev_num_kernels, num_kernels], stddev=0.1, seed=seed))
    b = tf.Variable(tf.constant(0.0, tf.float32, [num_kernels]))
    X = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    return X

def pool(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def relu(X):
    return tf.nn.relu(X)

def linear(X, num_weights, num_prev_weights):
    W = tf.Variable(tf.truncated_normal([num_prev_weights, num_weights], stddev=0.1, seed=seed))
    b = tf.Variable(tf.constant(0.1, tf.float32, [num_weights]))
    return tf.matmul(X, W) + b

def dropout(X):
    return tf.nn.dropout(X, keep_prob)

# image size: 32x32
X = images
X = relu(conv(X, kernel_sizes[0], num_kernels[0], 3))
X = relu(conv(X, kernel_sizes[1], num_kernels[1], num_kernels[0]))
X = pool(X)
# image size now: 16x16
X = dropout(X)
X = relu(conv(X, kernel_sizes[2], num_kernels[2], num_kernels[1]))
X = relu(conv(X, kernel_sizes[3], num_kernels[3], num_kernels[2]))
X = pool(X)
# image size now: 8x8
flat_size = 8*8*num_kernels[3]
if more_layers:
    X = dropout(X)
    X = relu(conv(X, kernel_sizes[4], num_kernels[4], num_kernels[3]))
    X = relu(conv(X, kernel_sizes[5], num_kernels[5], num_kernels[4]))
    X = relu(conv(X, kernel_sizes[6], num_kernels[6], num_kernels[5]))
    X = relu(conv(X, kernel_sizes[7], num_kernels[7], num_kernels[6]))
    X = pool(X)
    # image size now: 4x4
    flat_size = 4*4*num_kernels[7]

X = dropout(X)
X = tf.reshape(X, [-1, flat_size])
X = relu(linear(X, hidden_sizes[0], flat_size))
X = relu(linear(X, hidden_sizes[1], hidden_sizes[0]))
X = linear(X, num_labels, hidden_sizes[1])
Y = X

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, labels))
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train = optimizer.minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


accuracies = []
losses = []
for batch in range(num_batches):
    start_time = time.clock()
    batch_images, batch_labels = next_batch(batch_size)
    feed_dict = {images:batch_images, labels:batch_labels, keep_prob:dropout_keep_prob}
    sess.run([train], feed_dict=feed_dict)
    acc = sess.run(accuracy, feed_dict=feed_dict)
    delta_time = time.clock() - start_time

    if delta_time > 0.5:
        print("[%6d] Train accuracy: %f, %f milliseconds"%(batch,acc, delta_time*1000))
    
    if batch % 100 == 0:
        test_size = 1000
        batch_images = test_images[:test_size, :]
        batch_labels = test_labels[:test_size, :]
        feed_dict = {images:batch_images, labels:batch_labels, keep_prob:1.0}
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print("[%6d] Test  accuracy: %f <"%(batch,acc) + "-"*20)
