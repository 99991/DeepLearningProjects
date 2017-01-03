import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 1
# given 10 samples
data_size  = 10
# predict the next 500 samples
n_out      = 500
# with a memory of 50
lstm_size  = 200

path = "project02_task2.csv"
data = np.genfromtxt(path, dtype=float, delimiter=',', names=True)

#x = data['x']
y = data['y']

sequence = y

def make_batch():
    train_data = []
    train_target = []
    for _ in range(batch_size):
        offset = np.random.randint(len(sequence) - data_size - n_out)
        train_data.append(sequence[offset:offset + data_size])
        offset += data_size
        train_target.append(sequence[offset:offset + n_out])
    return train_data, train_target

data = tf.placeholder(tf.float32, [batch_size, data_size])
target = tf.placeholder(tf.float32, [batch_size, n_out])

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

state = lstm.zero_state(batch_size, tf.float32)

# output.shape = (batch_size, lstm_size)
output, state = lstm(data, state)

W = tf.Variable(tf.random_normal([lstm_size, n_out]))
b = tf.Variable(tf.constant(0.1, shape=[n_out]))

if 0:
    # Without LSTM Cell
    output = data
    W = tf.Variable(tf.random_normal([data_size, n_out]))

output = tf.matmul(output, W) + b

loss = tf.reduce_mean(tf.nn.l2_loss(output - target))

minimize = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    train_data, train_target = make_batch()
    feed_dict = {data: train_data, target: train_target}
    out, _, lss = sess.run([output, minimize, loss], feed_dict=feed_dict)
    print("%6d: %f"%(i, lss))

train_data = []
for _ in range(batch_size):
    train_data.append(sequence[-data_size:])
out = sess.run(output, feed_dict={data:train_data})

for i in range(batch_size):
    plt.plot(np.concatenate([sequence, out[i]]))
plt.show()

sess.close()
