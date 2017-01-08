import time
import cPickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib.layers import conv2d, bias_add, linear, batch_norm

def relu(X, leak=0.1):
    return tf.select(tf.less(X, 0.0), leak*X, X)

# variables
initial_learning_rate = 1.0
num_batches_for_decay = 2000
decay_rate            = 0.9
batch_size            = 64
num_batches           = 10001
kernel_size           = 3
num_kernels           = 32
num_hidden            = 500
depth                 = 3
drop_keep_prob        = 1.0
augment_images        = False
normalize_images      = False
optimizer = tf.train.AdamOptimizer()
learning_rate = tf.placeholder(tf.float32)

logDir                = './logs'
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# constants
image_width  = 32
image_height = 32
num_channels = 3
num_labels   = 10

cifar_directory = "../../cifar-10-batches-py"

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
with tf.name_scope("initialize_placeholders") as scope:
    images        = tf.placeholder(tf.float32, [None, image_width, image_height, num_channels])
    labels        = tf.placeholder(tf.float32, [None, num_labels])
    keep_prob     = tf.placeholder(tf.float32)
    is_training   = tf.placeholder(tf.bool)

def pool(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling')

def conv(X, num_kernels):
    return conv2d(X, num_kernels, kernel_size, activation_fn=None)

def flatten(X):
    flat_size = np.prod(X.get_shape().as_list()[1:])
    return tf.reshape(X, [-1, flat_size])

def per_image(image):
    image = tf.image.resize_image_with_crop_or_pad(image, image_height+4, image_width+4)
    image = tf.random_crop(image, [image_height, image_width, 3])
    image = tf.image.random_flip_left_right(image)
    return image

def bn(X):
    return batch_norm(X, updates_collections=None)

# Model
#######################################################
with tf.name_scope("start") as scope:
    X = images
    if augment_images:
        X = tf.select(is_training, tf.map_fn(per_image, X), X)
    if normalize_images:
        X = tf.map_fn(tf.image.per_image_standardization, X)

with tf.name_scope("single_convolution") as scope:
    X = conv(X, num_kernels)
with tf.name_scope("resnet") as scope:
    for i in range(depth):
        with tf.name_scope("single_resnet-layer") as scope:
            X = relu(bn(X))
            with tf.name_scope("convolution") as node:
                X = conv(X, num_kernels) + conv(bn(conv(X, num_kernels)), num_kernels)
            X = tf.nn.dropout(X, drop_keep_prob, name='dropout')
            with tf.name_scope("residual_convolution") as node:
                X += conv(relu(bn(conv(relu(bn(X)), num_kernels))), num_kernels)
            X = tf.nn.dropout(X, drop_keep_prob, name='dropout')
            X = pool(X)
with tf.name_scope("normalization") as node:
    X = bn(X)
#X = tf.reduce_mean(X, [1, 2])
with tf.name_scope("flattening") as node:
    X = flatten(X)
with tf.name_scope("hidden") as scope:
    with tf.name_scope('bias') as node:
        X = bias_add(X)
    with tf.name_scope('weights') as node:
        X = linear(X, num_hidden, activation_fn=relu, normalizer_fn=batch_norm)
with tf.name_scope("output-Layer") as scope:
    X = linear(X, num_labels, activation_fn=None)
    Y = X
#######################################################
with tf.name_scope('training'):
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, labels))
    tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(Y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('training-step'):
        train = optimizer.minimize(loss)
sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logDir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(logDir + '/test')
tf.global_variables_initializer().run()

train_steps = []
train_accuracies = []
train_losses = []
test_steps = []
test_accuracies = []
test_losses = []

def show_plots():
    plt.title("loss")
    plt.plot(train_steps, train_losses, label="train")
    plt.plot(test_steps, test_losses, label="test")
    plt.legend()
    plt.show()

    plt.title("accuracy")
    plt.plot(train_steps, train_accuracies, label="train")
    plt.plot(test_steps, test_accuracies, label="test")
    plt.legend()
    plt.show()

batch = 0
def step():
    global batch
    global initial_learning_rate
    
    if batch % num_batches_for_decay == 0:
        print("Learning rate set to %f"%initial_learning_rate)
        initial_learning_rate *= 1.0 - decay_rate
    
    start_time = time.clock()
    batch_images, batch_labels = next_batch(batch_size)
    feed_dict = {
        images:batch_images,
        labels:batch_labels,
        keep_prob:drop_keep_prob,
        is_training:True,
        learning_rate:initial_learning_rate}
    summary, _, acc, lss = sess.run([merged, train, accuracy, loss], feed_dict=feed_dict)
    train_writer.add_summary(summary, batch)
    train_accuracies.append(acc)
    train_losses.append(lss)
    train_steps.append(batch)
    delta_time = time.clock() - start_time

    # draw batch
    #if batch == 0: draw_images(batch_images, batch_labels)

    if delta_time > 0.3:
        print("[%6d] Train accuracy: %f, %f milliseconds"%(batch,acc, delta_time*1000))

    if batch % 100 == 0:
        test_size = 1000
        batch_images = test_images[:test_size, :]
        batch_labels = test_labels[:test_size, :]
        feed_dict = {
            images:batch_images,
            labels:batch_labels,
            keep_prob:1.0,
            is_training:False}
        summary, acc, lss = sess.run([merged, accuracy, loss], feed_dict=feed_dict)
        test_writer.add_summary(summary, batch)

        test_accuracies.append(acc)
        test_losses.append(lss)
        test_steps.append(batch)
        print("[%6d] Test  accuracy: %f <"%(batch,acc) + "-"*20)
    
    batch += 1

for _ in range(num_batches):
    step()

show_plots()
