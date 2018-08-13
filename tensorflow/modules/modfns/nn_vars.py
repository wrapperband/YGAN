"""
Instantiates TensorFlow variables for classes of networks
"""
import tensorflow as tf

##############################################################################
# GRU class

def gru_vars(obj, scope, name, embed_dim, latent_dim):
    """
    Sets TensorFlow variables for GRU as class attributes.

    Args:
        obj: class object (typically 'self')
        scope: name to give TF variable scope
        name: name to give class attribute
        embed_dim: GRU input dimension
        latent_dim: GRU hidden dimension
    """
    with tf.variable_scope(scope):
        # Update gate
        setattr(obj, name + '_W_ux',
                tf.get_variable(name="W_ux",
                                shape=[embed_dim, latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
        setattr(obj, name + '_W_uh',
                tf.get_variable(name="W_uh",
                                shape=[latent_dim, latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
        setattr(obj, name + '_b_u',
                tf.get_variable(name="b_u",
                                initializer=tf.ones(latent_dim)))
        # Forget gate
        setattr(obj, name + '_W_rx',
                tf.get_variable(name="W_rx",
                                shape=[embed_dim, latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
        setattr(obj, name + '_W_rh',
                tf.get_variable(name="W_rh",
                                shape=[latent_dim, latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
        setattr(obj, name + '_b_r',
                tf.get_variable(name="b_r",
                                initializer=tf.ones(latent_dim)))
        # Update function
        setattr(obj, name + '_W_hx',
                tf.get_variable(name="W_hx",
                                shape=[embed_dim, latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
        setattr(obj, name + '_W_hh',
                tf.get_variable(name="W_hh",
                                shape=[latent_dim, latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
        setattr(obj, name + '_b_h',
                tf.get_variable(name="b_h",
                                shape=[latent_dim],
                                initializer=tf.glorot_uniform_initializer()))
