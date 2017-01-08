import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import linear
from tensorflow.python.ops import rnn

path = "project02_task2.csv"
#path = "google.trends.urlaub.multiTimeline2004-2016.yearly.combined.csv"
data = np.genfromtxt(path, dtype=float, delimiter=',', names=True)

sequence = data['y']
n_test = 100
test_sequence = sequence[-n_test:]
sequence = sequence[:-n_test]

batch_size = 10
data_size = 200
lstm_size = 10

data = tf.placeholder(tf.float32, [None, data_size])
target = tf.placeholder(tf.float32, [None, 1])

data_list = tf.split(1, data_size, linear(data, data_size, activation_fn=None))
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
output = rnn.rnn(lstm, data_list, dtype=np.float32)[0][-1]
output = linear(output, 100, activation_fn=tf.nn.relu)
output = linear(output, 100, activation_fn=tf.nn.relu)
output = linear(output, 1, activation_fn=None)

loss = tf.reduce_mean(tf.nn.l2_loss(output - target))

minimize = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000 + 1):
    train_data = []
    train_target = []
    for _ in range(batch_size):
        offset = np.random.randint(len(sequence) - data_size - 1)
        train_data.append(sequence[offset:offset+data_size])
        train_target.append([sequence[offset+data_size]])
    
    feed_dict = {data: train_data, target: train_target}
    out, _, lss = sess.run([output, minimize, loss], feed_dict=feed_dict)
    if i % 100 == 0:
        print("%6d: %f"%(i, lss))

n = len(sequence)
sequence = list(sequence)
for _ in range(len(sequence)):
    train_data = [sequence[-data_size:]]
    out = sess.run(output, feed_dict={data:train_data})
    predicted = out[0, 0]
    sequence.append(predicted)

sess.close()

delta = np.abs(test_sequence - sequence[n:n+n_test])
mad = np.max(delta)
mse = np.sum(delta*delta)/len(delta)
print("Mean squared error: %f"%mse)
print("Maximum absolute difference: %f"%mad)

plt.plot(range(n), sequence[:n], label="given")
plt.plot(range(n-1, n*2), sequence[n-1:], label="predicted")
plt.legend(loc=2)
plt.show()
