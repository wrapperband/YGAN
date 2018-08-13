"""
The batch disciminator for the GeoGAN

@author: Robin N. Tully & Sean A. Cantrell
"""
import tensorflow as tf
from .basic_nets import Connected, Dense

class Batch_Dis():
    def __init__(self, state_dim, batch_size, hid_state, hid_batch):
        with tf.variable_scope('batch_reduce_1'):
            self.batch_reduce_1 = Connected(hid_batch, batch_size)
        with tf.variable_scope('batch_reduce_2'):
            self.batch_reduce_2 = Connected(hid_batch, hid_batch)
        with tf.variable_scope('batch_reduce_3'):
            self.batch_reduce_3 = Connected(1, hid_batch)

        with tf.variable_scope('state_reduce'):
            self.state_reduce = Connected(hid_state, state_dim)

        with tf.variable_scope('batch_result'):
            self.bin_class = Dense(1, hid_state, 3)

    def __call__(self, batches):
        batches = tf.transpose(batches)
        l1 = self.batch_reduce_1(batches)
        l2 = self.batch_reduce_2(l1)
        l3 = self.batch_reduce_3(l2)
        l3 = tf.transpose(l3)

        l4 = self.state_reduce(l3)
        out = self.bin_class(l4)
        return tf.squeeze(out)
