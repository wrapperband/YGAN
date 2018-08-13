"""
The disciminator for the GeoGAN
Placeholder for discriminator made by Robin Tully
"""
import tensorflow as tf
from .basic_nets import GeoGRU, dense

class Discriminator(GeoGRU):
    def __init__(self, state_dim, word_dim, class_no):
        super(Discriminator, self).__init__(state_dim, word_dim)
        with tf.variable_scope('reasoning'):
            self.reasoning = dense(class_no, state_dim, 10)

    def state_to_pdf(self, states):
        energy = self.reasoning(states)
        pdf = tf.nn.softmax(energy)
        return pdf

    def __call__(self, sents):
        states = self.encode(sents)
        energy = self.reasoning(states)
        pdf = tf.nn.softmax(energy)
        return pdf, states
