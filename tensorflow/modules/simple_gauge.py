"""
TensorFlow implementation of the Gauge Neural Network
@author: Sean A. Cantrell
"""

import tensorflow as tf
from .basic_nets import GRU, Dense, Connected, Gauge_evolver, Recon_dense
import numpy as np

##############################################################################
# Gauge Vector Class
##############################################################################

class Gauge:
    def __init__(self, state_dim, word_dim, latent_dim):
        self.state_dim = state_dim
        self.word_dim = word_dim
        self.latent_dim = latent_dim

        # Instantiate the encoding representations
        with tf.variable_scope("evolve_state_forw"):
            self.state_forw = GRU(state_dim, word_dim)
            self.state_back = self.state_forw

        # Instantiate the word inverter
        with tf.variable_scope("invert_word"):
            self.invert_word = Dense(word_dim, word_dim, 30)

        # Instantiate the decoding representation
        with tf.variable_scope("gen_word"):
            self.gauge_forw = Gauge_evolver(state_dim, word_dim, latent_dim)

        # Instantiate the boundary value functions
        with tf.variable_scope("word_bc"):
            self.word_bc = Dense(word_dim, 2 * state_dim, 20)

        # Assuming uniqueness of BC over all latent variables
        # Instantiate inverse BC
        with tf.variable_scope("state_reconstruction"):
            self.state_recon = Recon_dense(state_dim, latent_dim, 10)


    def iterate_fields(self, inputs):
        w0, h0, hbar0, z = inputs
        # Word and momentum
        w = self.gauge_forw(w0, h0, hbar0, z)
        # Forward + backward states
        h = self.state_forw.update_state(w0, h0)
        w0_inv = self.invert_word(w0)
        hbar = self.state_back.update_state(w0_inv, hbar0)
        return w, h, hbar, z

    def iter_schedule(self, inputs, skip):
        w0 = inputs[:,:self.word_dim]
        h0 = inputs[:,self.word_dim:self.word_dim + self.state_dim]
        hbar0 = inputs[:,self.word_dim + self.state_dim:self.word_dim + 2 * self.state_dim]
        z = inputs[:,self.word_dim + 2 * self.state_dim:]
        init = (w0, h0, hbar0, z)

        gen_iter = tf.range(skip)
        w_t, h_t, hbar_t, _ = tf.scan(lambda h, _: self.iterate_fields(h),
                                      gen_iter, init)
        outputs = tf.concat([w_t, h_t, hbar_t], axis=2)
        return outputs
        
    def decode(self, z, max_length):
        w0 = self.word_bc(z)
        hbar0 = z[:,self.state_dim:]
        h0 = tf.zeros_like(hbar0)
        init = (w0, h0, hbar0, z)
        gen_iter = tf.range(max_length-1)
        w_t, h_t, hbar_t, _ = tf.scan(lambda h, _: self.iterate_fields(h),
                                      gen_iter, init)
        _, h_fin, _, _ = self.iterate_fields((w_t[-1], h_t[-1], hbar_t[-1], z))
        
        # Words
        w0 = tf.expand_dims(w0, axis=1)
        words = tf.concat([w0, tf.transpose(w_t, perm=[1,0,2])], axis=1)
        # Forword states
        h = tf.concat([tf.transpose(h_t, perm=[1,0,2]),
                       tf.expand_dims(h_fin, axis=1)], axis=1)
        # Reverse states
        hbar0 = tf.expand_dims(hbar0, axis=1)
        hbar_r = tf.concat([hbar0, tf.transpose(hbar_t, perm=[1,0,2])], axis=1)
        hbar = tf.reverse(hbar_r, axis=[1])
        return words, h, hbar

    def autoencode(self, real):
        max_length = real.shape.as_list()[1]

        # Encode the inputs
        states_forw, states_back = self.get_states(real)
        encoding = tf.concat([states_forw, states_back], axis=-1)
    
        # Reconstruct latent variable
        z = encoding[:,-1]
    
        # Generate synthetic sentence
        synth, h, hbar = self.decode(z, max_length)
        states = tf.concat([h, hbar], axis=-1)
        return synth, states, encoding

    def schedule(self, real, skip):
        shape = real.shape.as_list()
        max_length = shape[1]
        batch_size = shape[0]
        skip_no = int(np.ceil((max_length-1)/skip))

        # Encode the inputs
        states_forw, states_back = self.get_states(real)
        encoding = tf.concat([states_forw, states_back], axis=-1)
        z = encoding[:,-1]

        # Generate synthetic sentence with scheduling
        w0 = self.word_bc(z)
        hbar0 = states_back[:,-1]
        h0 = tf.zeros_like(hbar0)

        # Schedule iterator
        w_iter = tf.transpose(real[:,:-1], perm=[1,0,2])
        h_iter = tf.transpose(tf.concat([tf.expand_dims(h0, axis=1),
                              states_forw[:,:-2]], axis=1), perm=[1,0,2])
        hbar_iter = tf.transpose(tf.reverse(states_back, axis=[1])[:,:-1],
                                 perm=[1,0,2])
        z_iter = tf.transpose(tf.reshape(tf.tile(z, [1,max_length-1]),
                              shape=[-1, max_length-1, 2 * self.state_dim]),
                              perm=[1,0,2])
        schedule_iter_full = tf.concat([w_iter, h_iter, hbar_iter, z_iter],
                                       axis=-1)
        schedule_iter = schedule_iter_full[::skip]

        # Scheduled predictions
        pred_raw = tf.map_fn(lambda x: self.iter_schedule(x, skip),
                             schedule_iter)
        pred_t = tf.reshape(pred_raw, [skip_no * skip, batch_size, -1])[:max_length - 1]
        pred = tf.transpose(pred_t, perm=[1,0,2])
        w_pred = pred[:,:,:self.word_dim]
        h_pred = pred[:,:,self.word_dim:self.word_dim + self.state_dim]
        hbar_pred = pred[:,:,self.word_dim + self.state_dim:]
        _, h_pred_fin, _, _ = self.iterate_fields((w_pred[:,-1], h_pred[:,-1], hbar_pred[:,-1], z))

        w = tf.concat([tf.expand_dims(w0, axis=1), w_pred], axis=1)
        h = tf.concat([h_pred, tf.expand_dims(h_pred_fin, axis=1)], axis=1)
        hbar = tf.concat([tf.expand_dims(hbar0, axis=1), hbar_pred], axis=1)

        return w, h, tf.reverse(hbar, axis=[1])
        

    def get_states(self, sent):
        self.state_forw(sent)
        states_forw = self.state_forw.h_set
        sent_back = tf.reverse(sent, axis=[1])
        self.state_back(sent_back)
        states_back = self.state_back.h_set
        return states_forw, states_back

    def get_and_concat_states(self, sent):
        forw, back = self.get_states(sent)
        return tf.concat([forw, tf.reverse(back, axis=[1])], axis=-1)
