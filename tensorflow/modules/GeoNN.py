"""
TensorFlow implementation of the Geodesic Neural Network
@author: Sean A. Cantrell
"""
import tensorflow as tf
from .basic_nets import GRU, dense
import numpy as np

##############################################################################
# Generative Geodesic class
##############################################################################

class GeoNN:
    def __init__(self, state_dim, word_dim):
        self.state_dim = state_dim
        self.word_dim = word_dim

        # Instantiate the decoding RNNs
        with tf.variable_scope("momentum"):
            self.R_p = GRU(state_dim, state_dim)
        with tf.variable_scope("word"):
            self.R_w = GRU(state_dim, state_dim)
#            initializer = tf.random_normal_initializer(1., 0.1)
#            self.projector = tf.get_variable(name='projector',
#                                             shape=[state_dim, word_dim],
#                                             initializer=initializer)
        with tf.variable_scope("coordinate_change"):
            self.P = dense(word_dim, state_dim, 10)
        

        # Instantiate the initial conditions
        # This is just for declarative purposes
        self.p0 = None
        self.w0 = None

        # Instantiate states
        # This is just for declarative purposes
        self.sentence = None
        self.momenta = None

        # Instantiate collection of TF variables
        self.vars_list = (tf.trainable_variables(scope='momentum') +
                          tf.trainable_variables(scope='word'))
        self.vars_vals = None

    def iter_p(self, p):
        """
        Shorter wrapper for the momentum update function
        """
        return self.R_p.update_state(p, p)

    def iter_w(self, p, w):
        """
        Shorter wrapper for the word update function
        """
        return self.R_w.update_state(p, w)

    def get_next_word(self, state):
        """
        Iterates momentum and word
        """
        p0, w0 = state
        p1 = self.iter_p(p0)
        w1 = self.iter_w(p1, w0)
        return (p1, w1)

    def gen(self, p0, w0, length):
        """
        Decoding explicitly for training.
        When inferring, specifying length should be unnecessary.
        Prior to training, the network cannot know when to issue a halting
        word.
        Consequently, we use the batched sentence length to control output
        during training.

        >>> tf.reset_default_graph()
        >>> geo = GeoNN(5)
        >>> p0 = tf.random_uniform(shape=[3,5])
        >>> w0 = tf.random_uniform(shape=[3,5])
        >>> gen = geo.gen(p0, w0, 10)
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> print(sess.run(gen).shape)
        (3, 10, 5)
        """
        state0 = (p0, w0)
        states = tf.scan(lambda h, x: self.get_next_word(h),
                         tf.range(length),
                         state0)
#        words = tf.einsum('ijk,kl->ijl', states[1], self.projector)
        words = tf.map_fn(lambda x: self.P(x), states[1])
        self.sentence = tf.transpose(words, perm=[1,0,2])
        return self.sentence
    
    def __call__(self, p0, w0):
        """
        Non-training generation.
        Assumes batch generation.
        Uses numpy initializers.
        """
        p0 = np.float32(p0)
        w0 = np.float32(w0)
        states = np.array([[p0, w0]])

        # Define the condition and body functions to execute in a while loop
        def condit(states):
            x = states[-1,1]
            dim = x.shape[-1]
            final_token = np.sqrt(np.max(np.sum(x**2, axis=1)))
            cond = final_token >= 1e-3 * np.sqrt(dim)
            return cond

        def body(states, sess):
            p1, w1 = sess.run(self.get_next_word(states[-1]))
            next_state = np.float32(np.array([[p1, w1]]))
            new_states = np.concatenate([states, next_state], axis=0)
            return new_states

        # Generate the sentences in a while loop
        def gen_sentences(states, sess):
            cond = True
            while cond:
                states = body(states, sess)
                cond = condit(states)
            states = sess.run(tf.einsum('ijkl,lm->ijkm',
                                        states,
                                        self.projector))
            return states

        with tf.Session() as sess:
            # Assign model parameters stored values
            for i, val in enumerate(self.vars_vals):
                sess.run(self.vars_list[i].assign(val))
            final_states = np.transpose(gen_sentences(states, sess),
                                        axes=[1,2,0,3])
        self.sentence = final_states[1]
        self.momenta = final_states[0]

        return (self.momenta, self.sentence)
