import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import linear, bias_add, batch_norm

batch_size = 10
# given some number of sample sequences
data_size  = 10
lstm_size  = 4

path = "project02_task2.csv"
#path = "google.trends.urlaub.multiTimeline2004-2016.yearly.combined.csv"
data = np.genfromtxt(path, dtype=float, delimiter=',', names=True)

#x = data['x']
y = data['y']
y -= np.mean(y)
y /= np.max(np.abs(y))

sequence = y
#sequence += np.linspace(0, 100, len(y))

def make_batch():
    train_data = []
    train_target = []
    for _ in range(batch_size):
        offset = np.random.randint(len(sequence) - data_size - 1)
        train_data.append(sequence[offset:offset + data_size])
        offset += data_size
        train_target.append(sequence[offset:offset + 1])
    return train_data, train_target

data = tf.placeholder(tf.float32, [batch_size, data_size])
target = tf.placeholder(tf.float32, [batch_size, 1])

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
