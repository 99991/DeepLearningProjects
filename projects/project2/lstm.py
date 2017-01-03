import numpy as np
import tensorflow as tf

batch_size = 3
data_size = 4
lstm_size = 5

# output: shape = (batch_size, lstm_size)
# state: (c, h)
# c: shape = (batch_size, lstm_size)
# h: shape = (batch_size, lstm_size)

data = tf.placeholder(tf.float32, [batch_size, data_size])
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

state = lstm.zero_state(batch_size, tf.float32)

output, state = lstm(data, state)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_data = np.random.rand(batch_size, data_size).astype(np.float32)

feed_dict = {data: train_data}
output, state = sess.run([output, state], feed_dict=feed_dict)

print("Output:")
print(output.shape)
print("")
print("State:")
print(state.c.shape)
print(state.h.shape)

sess.close()
