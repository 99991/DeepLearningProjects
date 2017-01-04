import time
import cPickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib.layers import conv2d, bias_add, linear, batch_norm
import augment

# variables
learning_rate     = 0.001
batch_size        = 64
num_batches       = 10001
kernel_size       = 3
num_kernels       = 16
num_hidden        = 500
drop_keep_prob    = 1.0
augment_images    = False
weird_net         = False

# constants
image_width  = 32
image_height = 32
num_channels = 3
num_labels   = 10

cifar_directory = "../../cifar-10-batches-py"

# TODO
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

def draw_images(images, labels, offset=0):
    # find divisible number close to square
    n = len(images)
    nx = int(np.sqrt(n))
    while n % nx != 0:
        nx += 1
    ny = n // nx
    
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
images      = tf.placeholder(tf.float32, [None, image_width, image_height, num_channels])
labels      = tf.placeholder(tf.float32, [None, num_labels])
keep_prob   = tf.placeholder(tf.float32)

def pool(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def relu(X, leak=0.1):
    return tf.select(tf.less(X, 0.0), leak*X, X)

def conv(X):
    X = conv2d(relu(X), num_kernels, kernel_size, activation_fn=None, normalizer_fn=batch_norm)
    X = tf.nn.dropout(X, keep_prob)
    X = bias_add(X)
    X = tf.nn.local_response_normalization(X)
    return X

def flatten(X):
    flat_size = np.prod(X.get_shape().as_list()[1:])
    return tf.reshape(X, [-1, flat_size])
    
# Model
#######################################################
X = images
X = conv(X)
if weird_net:
    all_X = []
    all_X.append(X)
    X += conv(X)
    all_X.append(X)
    X = pool(X)
    all_X.append(X)
    X += conv(X)
    all_X.append(X)
    X += conv(X)
    all_X.append(X)
    X = pool(X)
    all_X.append(X)
    X += conv(X)
    all_X.append(X)
    X = tf.concat(1, map(flatten, all_X))
else:
    for _ in range(20):
        X += conv(X)
    X = pool(X)
    for _ in range(5):
        X += conv(X)
    X = pool(X)
    X = flatten(X)

X = bias_add(X)

X = linear(X, num_hidden, activation_fn=relu, normalizer_fn=batch_norm)
X = linear(X, num_hidden, activation_fn=relu, normalizer_fn=batch_norm)
X = linear(X, num_hidden, activation_fn=relu, normalizer_fn=batch_norm)

X = linear(X, num_labels, activation_fn=None)
Y = X
#######################################################

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, labels))
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def get_augmented_images(images):
    new_images = []
    for i in range(len(images)):
        image = images[i]
        image = augment.get_augmented_image(image)
        new_images.append(image)
    return new_images

accuracies = []
losses = []
for batch in range(num_batches):
    start_time = time.clock()
    batch_images, batch_labels = next_batch(batch_size)
    if augment_images:
        batch_images = get_augmented_images(batch_images)
    feed_dict = {images:batch_images, labels:batch_labels, keep_prob:drop_keep_prob}
    _, acc = sess.run([train, accuracy], feed_dict=feed_dict)
    delta_time = time.clock() - start_time

    # draw batch
    #if batch == 0: draw_images(batch_images, batch_labels)

    if delta_time > 0.3:
        print("[%6d] Train accuracy: %f, %f milliseconds"%(batch,acc, delta_time*1000))
    
    if batch % 100 == 0:
        test_size = 1000
        batch_images = test_images[:test_size, :]
        batch_labels = test_labels[:test_size, :]
        feed_dict = {images:batch_images, labels:batch_labels, keep_prob:1.0}
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print("[%6d] Test  accuracy: %f <"%(batch,acc) + "-"*20)
