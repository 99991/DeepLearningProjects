import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import linear, bias_add, batch_norm

path = "project02_task2.csv"
#path = "google.trends.urlaub.multiTimeline2004-2016.yearly.combined.csv"
data = np.genfromtxt(path, dtype=float, delimiter=',', names=True)

y = data['y']
y -= np.mean(y)
y /= np.max(np.abs(y))

sequence = y

data_size = 10

data = tf.placeholder(tf.float32, [data_size, 1])

target = tf.placeholder(tf.float32, [1])

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

state = lstm.zero_state(batch_size, tf.float32)

# output.shape = (batch_size, lstm_size)
output, state = lstm(data, state)

X = output
#X = data
for _ in range(1):
    X = linear(X, 100, activation_fn=tf.nn.relu)
X = linear(X, 1, activation_fn=None)
output = X

loss = tf.reduce_mean(tf.nn.l2_loss(output - target))

minimize = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    train_data, train_target = make_batch()
    feed_dict = {data: train_data, target: train_target}
    out, _, lss = sess.run([output, minimize, loss], feed_dict=feed_dict)
    if i % 100 == 0:
        print("%6d: %f"%(i, lss))

sequence = list(sequence)
n = len(sequence)
for _ in range(n):
    train_data = [sequence[-data_size:]]*batch_size
    out = sess.run(output, feed_dict={data:train_data})
    predicted = out[0, 0]
    sequence.append(predicted)

plt.plot(range(n), sequence[:n], label="given")
plt.plot(range(n-1, n*2), sequence[n-1:], label="predicted")
plt.legend(loc=2)
plt.show()

sess.close()
