import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.layers import linear
import matplotlib.pyplot as plt

# make very simple sequence
sequence = list(np.cos(np.linspace(0, 2*np.pi, 50))*0.5)
#sequence = [-0.5, 0.5]*3

# sequence placeholder node (has to be 2d for some reason)
sequence_placeholder = tf.placeholder(tf.float32, [1, len(sequence)])
target_placeholder   = tf.placeholder(tf.float32, [1, len(sequence)])

# split sequence into list of values
sequence_list = tf.split(1, len(sequence), sequence_placeholder)
target_list   = tf.split(1, len(sequence),   target_placeholder)

lstm_size = 10
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
stacked = tf.nn.rnn_cell.MultiRNNCell([lstm]*3)
# apply lstm cell to each value of sequence
output_list, unused_states = rnn.rnn(lstm, sequence_list, dtype=np.float32)

"""
loss = 0
for output, target in zip(output_list, target_list):
    loss += tf.nn.l2_loss(output - target)
"""
output = tf.stack(output_list)
output = tf.squeeze(output)
output = linear(output, 10)
output = linear(output, 1)
output = tf.reshape(output, [1, len(sequence)])
loss = tf.nn.l2_loss(output - target_placeholder)

minimize = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def make_shifted():
    shift = 7
    return sequence[-shift:] + sequence[:-shift]

for i in range(1000 + 1):
    # make target sequence by shifting sequence
    shifted_sequence = make_shifted()
    
    # train to predict shifted sequence from sequence
    _, loss_result = sess.run(
        [minimize, loss],
        feed_dict={
            sequence_placeholder:[sequence],
            target_placeholder:[shifted_sequence]})

    # print loss from time to time
    if i % 100 == 0:
        print(i, loss_result)
    
    sequence = shifted_sequence

# predict shifted sequence
predicted = sess.run([output], feed_dict={sequence_placeholder:[sequence]})
print("Predicted:", predicted)
print(np.array(predicted) - make_shifted())

plt.plot(predicted[0][0], label="predicted")
plt.plot(make_shifted(), label="shifted")
plt.plot(sequence, label="original")
plt.legend()
plt.show()
