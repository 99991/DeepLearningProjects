import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt

def run_nn(**args):
	print("Arguments:")
	for key, val in args.items():
			print(key, val)

	# add arguments to local namespace
	batch_size = args['batch_size']
	num_kernels1 = args['num_kernels1']
	num_hidden = args['num_hidden']
	regularization_factor = args['regularization_factor']
	dropout_keep_probability = args['dropout_keep_probability']
	learning_rate = args['learning_rate']
	kernel1_size = args['kernel1_size']
	test_interval = args['test_interval']
	num_batches = args['num_batches']
	seed = args['seed']
	pool = args['pool']
	
	# make runs deterministic
	np.random.seed(seed)

	mnist = read_data_sets("../../mnist", one_hot=True)
		
	image_size = 28
	num_labels = 10

	# placeholders for training and testing data
	data          = tf.placeholder(tf.float32, [None, image_size**2])
	results       = tf.placeholder(tf.float32, [None, num_labels])
	keep_prob     = tf.placeholder(tf.float32)
	
	# those two optimizers work okish with learning rate of 0.1
	#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)

	flat_size = (image_size/pool)**2*num_kernels1

	conv1_weights = tf.Variable(tf.truncated_normal([kernel1_size, kernel1_size, 1, num_kernels1], stddev=0.1, seed=seed))
	conv1_biases  = tf.Variable(tf.constant(0.0, tf.float32, [num_kernels1]))
	fc1_weights   = tf.Variable(tf.truncated_normal([flat_size, num_hidden], stddev=0.1, seed=seed))
	fc1_biases    = tf.Variable(tf.constant(0.1, tf.float32, [num_hidden]))
	fc2_weights   = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1, seed=seed))
	fc2_biases    = tf.Variable(tf.constant(0.1, tf.float32, [num_labels]))

	X = data
	# reshape data to image shape
	#X = tf.nn.dropout(X, keep_prob)
	X = tf.reshape(X, [-1, image_size, image_size, 1])
	# conv relu pool
	X = tf.nn.conv2d(X, conv1_weights, strides=[1,1,1,1], padding='SAME')
	X = tf.nn.relu(tf.nn.bias_add(X, conv1_biases))
	X = tf.nn.max_pool(X, ksize=[1,pool,pool,1], strides=[1,pool,pool,1], padding='SAME')
	# flatten data to row shape
	X = tf.reshape(X, [-1, flat_size])
	# relu(XW + b)
	X = tf.nn.relu(tf.matmul(X, fc1_weights) + fc1_biases)
	# dropout
	X = tf.nn.dropout(X, keep_prob, seed=seed)
	# XW + b
	X = tf.matmul(X, fc2_weights) + fc2_biases
	Y = X

	# regularization to keep weights small
	regularization = (
		tf.nn.l2_loss(fc1_weights) +
		tf.nn.l2_loss(fc2_weights) +
		tf.nn.l2_loss(fc1_biases ) +
		tf.nn.l2_loss(fc2_biases ))

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, results))
	loss += regularization_factor*regularization
	correct_prediction = tf.equal(tf.argmax(results, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	train = optimizer.minimize(loss)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	accuracies = []
	losses = []
	for batch in range(num_batches):
		batch_data, batch_results = mnist.train.next_batch(batch_size)
		feed_dict = {data:batch_data, results:batch_results, keep_prob:dropout_keep_probability}
		sess.run([train], feed_dict=feed_dict)
		
		if batch % test_interval == 0:
			feed_dict = {data:mnist.test.images, results:mnist.test.labels, keep_prob:1.0}
			weights, acc, lss, predicted = sess.run(
								[conv1_weights, accuracy, loss, tf.argmax(Y, 1)],
								feed_dict=feed_dict)
			print("batch %5d, accuracy %f, loss %f"%(batch, acc, lss))
			
			accuracies.append(acc)
			losses.append(lss)
	
	return accuracies, losses, weights, predicted, mnist.test.images, np.argmax(mnist.test.labels, 1)

args = {
	"batch_size":                512,
	"num_kernels1":               16,
	"num_hidden":               1024,
	"regularization_factor":    1e-4,
	"dropout_keep_probability":  0.5,
	"learning_rate":           0.001,
	"kernel1_size":                5,
	"test_interval":             100,
	"num_batches":              2001,
	"seed":                      666,
	"pool":                        4,
}

_, _, _, predicted, images, labels = run_nn(**args)

mask = np.not_equal(predicted, labels)
indices = np.array(range(len(labels)))
indices_wrong_predictions = indices[mask]

i = 0
for y in range(8):
        for x in range(4):
                index = indices_wrong_predictions[i]
                i += 1
                ax = plt.subplot(4, 8, i)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                image = images[index, :]
                image = image.reshape((28, 28))
                plt.title(str(predicted[index]))
                plt.imshow(image, cmap='gray')
plt.tight_layout()
plt.show()
