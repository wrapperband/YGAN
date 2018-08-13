"""
@author: Robin N. Tully
"""

import basic_nets as bn

class Decoder:
    def __init__(self, state_dim, word_dim, latent_dim):
        """
        Contains forward encoding and backward encoding and a decoder
        
        Encoding takes a sentence and returns a state or series of states
        
        Latent reconstruction takes the final forward state and final back state
            and creates a latent variable
            
        Boundary conditions (abbreviated bc) take a latent variable and create
            the initial conditions to generate a sentence
            
        tf.scan(lambda h, x: state_update(x,h), x_series, h_init) nests successive
            applications of the state updating function, sampling successive elements
            of x_series.  The initial state is specified by h_init
        
        """
        with tf.variable_scope('encoder'):
            self.encoder = GRU(state_dim, word_dim)
        with tf.variable_scope('decoder'):
            self.decoder = GRU(word_dim, 2 * state_dim)
        with tf.variable_scope("latent_reconstruction"):
            self.latent_recon = Dense(latent_dim, 2 * state_dim, 10)
        # Instantiate the boundary value functions
        with tf.variable_scope("word_bc"):
            self.word_bc = Gauge_dense(word_dim, latent_dim, 10)
        with tf.variable_scope("state_bc"):
            self.state_back_bc = Dense(state_dim, latent_dim, 10)
    
        
    def decode(self):
        do the shit from rnn_modules in dale
        
    def autoencode(self, real):
        forw, back = self.get_states(real)
        encoding = tf.concat([forw, back], axis=-1)
        z = self.latent_recon(encoding)
        
        
    def encode(self, sent):
        forw = self.encoder(sent)
        back = self.encoder(tf.reverse(sent, axis=[1]))
        return forw, back
        
    def schedule:
        