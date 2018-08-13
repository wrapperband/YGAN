"""
Returns outputs from common functions in neural net classes
"""
#pylint: disable=R0914
import tensorflow as tf

##############################################################################
# GRU class

def gru_fn(obj, name, x, h):
    """
    Generic GRU function.

    Notation:
        Any variable name containing 'W' is a weight matrix;
        any variable name containing 'b' is a bias.

        The first subscript in a variable name refers to the function it
        helps; the second subcript refers to the argument it acts on.

        u = update gate
        r = forget gate
        h = state update function

    Args:
        obj: class object (typically 'self')
        name: name of class attribute (e.g. self.name)
        x: GRU input; a rank 2 matrix
        h: GRU hidden state; a rank 2 matrix
    Returns:
        A rank 2 tensor (updated hidden state matrix)
    """
    # Get variables
    W_ux = getattr(obj, name + '_W_ux')
    W_uh = getattr(obj, name + '_W_uh')
    b_u = getattr(obj, name + '_b_u')

    W_rx = getattr(obj, name + '_W_rx')
    W_rh = getattr(obj, name + '_W_rh')
    b_r = getattr(obj, name + '_b_r')

    W_hx = getattr(obj, name + '_W_hx')
    W_hh = getattr(obj, name + '_W_hh')
    b_h = getattr(obj, name + '_b_h')

    # Update gate
    u = tf.sigmoid(tf.matmul(x, W_ux) + tf.matmul(h, W_uh) + b_u)

    # 'Forget' gate
    r = tf.sigmoid(tf.matmul(x, W_rx) + tf.matmul(h, W_rh) + b_r)

    # Hidden state
    hp = tf.tanh(tf.matmul(x, W_hx) + r * tf.matmul(h, W_hh) + b_h)

    return (1 - u) * hp + u * h

def leak_gru_fn(obj, name, x, h):
    """
    Generic GRU function.

    Notation:
        Any variable name containing 'W' is a weight matrix;
        any variable name containing 'b' is a bias.

        The first subscript in a variable name refers to the function it
        helps; the second subcript refers to the argument it acts on.

        u = update gate
        r = forget gate
        h = state update function

    Args:
        obj: class object (typically 'self')
        name: name of class attribute (e.g. self.name)
        x: GRU input; a rank 2 matrix
        h: GRU hidden state; a rank 2 matrix
    Returns:
        A rank 2 tensor (updated hidden state matrix)
    """
    # Get variables
    W_ux = getattr(obj, name + '_W_ux')
    W_uh = getattr(obj, name + '_W_uh')
    b_u = getattr(obj, name + '_b_u')

    W_rx = getattr(obj, name + '_W_rx')
    W_rh = getattr(obj, name + '_W_rh')
    b_r = getattr(obj, name + '_b_r')

    W_hx = getattr(obj, name + '_W_hx')
    W_hh = getattr(obj, name + '_W_hh')
    b_h = getattr(obj, name + '_b_h')

    # Update gate
    u = tf.sigmoid(tf.matmul(x, W_ux) + tf.matmul(h, W_uh) + b_u)

    # 'Forget' gate
    r = tf.sigmoid(tf.matmul(x, W_rx) + tf.matmul(h, W_rh) + b_r)

    # Hidden state
    hp = tf.sigmoid(tf.matmul(x, W_hx) + r * tf.matmul(h, W_hh) + b_h)

    return (1 - u) * hp + u * h
