"""
TensorFlow implementation of the Gauge Neural Network
@author: Sean A. Cantrell
"""

import tensorflow as tf
from .basic_nets import GRU, Dense, Gauge_dense, Gauge_evolver
import numpy as np

##############################################################################
# Gauge Vector Class
##############################################################################

class GaugeNN:
    def __init__(self, state_dim, word_dim, latent_dim):
        self.state_dim = state_dim
        self.word_dim = word_dim
        self.latent_dim = latent_dim

        # Instantiate the encoding representations
        with tf.variable_scope("evolve_state"):
            self.state_forw = GRU(state_dim, word_dim)

        # Instantiate the word inverter
        with tf.variable_scope("invert_word"):
            self.invert_word = Dense(word_dim, word_dim, 10)

        # Instantiate the decoding representation
        with tf.variable_scope("gen_word"):
            self.gauge_forw = Gauge_evolver(state_dim, word_dim, latent_dim)

        # Instantiate the boundary value functions
        with tf.variable_scope("word_bc"):
            self.word_bc = Gauge_dense(word_dim, latent_dim, 10)
        with tf.variable_scope("state_bc"):
            self.state_back_bc = Dense(state_dim, latent_dim, 10)

        # Assuming uniqueness of BC over all latent variables
        # Instantiate inverse BC
        with tf.variable_scope("latent_reconstruction"):
            self.latent_recon = Dense(latent_dim, 2 * state_dim, 10)


    def iterate_fields(self, inputs):
        w0, h0, hbar0, z = inputs
        # Word and momentum
        w = self.gauge_forw(w0, h0, hbar0, z)
        # Forward + backward states
        h = self.state_forw.update_state(w0[:,0], h0)
        w0_inv = self.invert_word(w0[:,0])
        hbar = self.state_forw.update_state(w0_inv, hbar0)
        return w, h, hbar, z

    def gauge_schedule(self, inputs, gauges):
        words = tf.expand_dims(inputs[:,:self.word_dim], axis=1)
        h0 = inputs[:,self.word_dim:self.word_dim + self.state_dim]
        hbar0 = inputs[:,self.word_dim + self.state_dim:self.word_dim + 2 * self.state_dim]
        z = inputs[:,self.word_dim + 2 * self.state_dim:]
        w0 = tf.concat([words, gauges], axis=1)
        w = self.gauge_forw(w0, h0, hbar0, z)
        h = 
        
        
        

    def decode(self, z, max_length):
        w0 = self.word_bc(z)
        hbar0 = self.state_back_bc(z)
        h0 = tf.zeros_like(hbar0)
        init = (w0, h0, hbar0, z)
        gen_iter = tf.range(max_length-1)
        w_t, h_t, hbar_t, _ = tf.scan(lambda h, _: self.iterate_fields(h),
                                      gen_iter, init)
        # Words
        w0 = tf.expand_dims(w0, axis=1)
        w = tf.concat([w0, tf.transpose(w_t, perm=[1,0,2,3])], axis=1)
        words = w[:,:,0]
        # Forword states
        h = tf.concat([h0, tf.transpose(h_t, perm=[1,0,2])], axis=1)
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
        z = self.latent_recon(encoding[:,-1])
    
        # Generate synthetic sentence
        synth, h, hbar = self.decode(z, max_length)
        states = tf.concat([h, hbar], axis=-1)
        return synth, states, encoding

    def schedule(self, real):
        max_length = real.shape.as_list()[1]

        # Encode the inputs
        states_forw, states_back = self.get_states(real)
        encoding = tf.concat([states_forw, states_back], axis=-1)

        # Reconstruct latent variable
        z = self.latent_recon(encoding[:,-1])

        # Generate synthetic sentence with scheduling
        w0 = self.word_bc(z)
        hbar0 = self.state_back_bc(z)
        h0 = tf.zeros_like(hbar0)
        
        tf.map_fn(lambda x: stuff )

    def get_states(self, sent):
        self.state_forw(sent)
        states_forw = self.state_forw.h_set
        sent_back = tf.reverse(sent, axis=[1])
        self.state_forw(sent_back)
        states_back = self.state_back.h_set
#        states = tf.concat([states_forw, states_back], axis=-1)
        return states_forw, states_back