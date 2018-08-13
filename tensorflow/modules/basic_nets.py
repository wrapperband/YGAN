"""
TensorFlow implementation of
RNN and MLP layer classes for encoding and decoding
@author: Sean A. Cantrell
"""
#pylint: disable=F0401,R0902,C0103
import tensorflow as tf
from .modfns.nn_vars import gru_vars as gv
from .modfns.nn_fns import gru_fn as gf
from .modfns.nn_fns import leak_gru_fn as lgf

##############################################################################
# GRU class
##############################################################################

class GRU:
    """
    Constructs a GRU-based encoder, with modifications for initial conditions
    and style of output.

    Notation:
        Any variable name containing 'W' is a weight matrix;
        any variable name containing 'b' is a bias.

        The first subscript in a variable name refers to the function it
        helps; the second subcript refers to the argument it acts on.

        u = update gate
        r = forget gate
        h = state update function

    Args:
        embed_dim: dimension of the inputs into the GRU
        latent_dim: hidden dimension output from the GRU

    The memory vector in the decoder is the dimension of the inputs.

    >>> tf.reset_default_graph()
    >>> rnn = GRU(4, 6)
    """
    def __init__(self, state_dim, embed_dim):
        self.state_dim = state_dim  # Dimension of the hidden layer
        self.embed_dim = embed_dim  # Dimension of the input layer
        # Declare the TensorFlow variables
        gv(self, 'state_update', 'state', embed_dim, state_dim)

        # Declare properties that class functions will assign values to
        self.h = None
        self.h_set = None

    def update_state(self, x, h):
        """
        Takes an input vector and state and returns an updated state according
        to GRU logic:
            u = sigma(W_{ux} x + W_{uh} h0 + b_u)
            r = sigma(W_{rx} x + W_{uh} h0 + b_r)
            h' = tanh(W_{hx} x + r*W_{hh} h + b_h)
            h1 = (1-u)*h' + u*h
        where * is the Hadamard product,
        normal multiplication is implied matrix multiplication.

        Args:
            x: Rank 2 tensor of size [batch_size]*[embed_dim];
                this is the word used to update the hidden state
            h: Rank 2 tensor of size [batch_size]*[latent_dim];
                this is the current hidden state
        Returns:
            A rank 2 tensor of size [batch_size]*[latent_dim];
            this is the updated hidden state

        >>> tf.reset_default_graph()
        >>> rnn = GeoGRU(4,6)
        >>> word = tf.random_uniform([10,6])
        >>> state0 = tf.random_uniform([10,4])
        >>> state1 = rnn.update_state(word,state0)
        >>> shape = tf.shape(state1)
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> sess.run(shape)
        array([10,  4], dtype=int32)
        """
        new_state = gf(self, 'state', x, h)

        return new_state

    def encode(self, x):
        """
        Iterates over word vectors to return a hidden state encoding.

        Args:
            x: A rank 3 tensor of size [batch_size]*[token_num]*[embed_dim]
        Returns:
            A rank 2 tensors of size [batch_size]*[latent_dim]
        Additional:
            Adds sequence of hidden states constructed while encoding as
            property of the gru class.

        >>> tf.reset_default_graph()
        >>> sent = tf.random_uniform([10,8,6])
        >>> rnn = GeoGRU(4,6)
        >>> encoding = rnn.encode(sent)
        >>> shapes = [tf.shape(encoding), tf.shape(rnn.h_set)]
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> sess.run(shapes)
        [array([10,  4], dtype=int32), array([10,  8,  4], dtype=int32)]
        """
        init_state = tf.zeros(shape=[tf.shape(x)[0], self.state_dim])
        x_t = tf.transpose(x, perm=[1, 0, 2])
        h_set_t = tf.scan(lambda h, x: self.update_state(x, h),
                          x_t, init_state)
        h_set = tf.transpose(h_set_t, perm=[1, 0, 2])

        # The sequence of hidden states constructed while encoding
        self.h_set = h_set
        # The final encoding of the sequence of embedding vectors
        self.h = h_set[:, -1]

        return self.h

    def __call__(self, x):
        """
        Simply calls gru.encode(x).

        Args:
            x: A rank 3 tensor of size [batch_size]*[token_num]*[embed_dim]
        Returns:
            A rank 2 tensors of size [batch_size]*[latent_dim]

        >>> tf.reset_default_graph()
        >>> sent = tf.random_uniform([10,8,6])
        >>> rnn = GeoGRU(4,6)
        >>> encoding = rnn(sent)
        >>> shapes = [tf.shape(encoding), tf.shape(rnn.h_set)]
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> sess.run(shapes)
        [array([10,  4], dtype=int32), array([10,  8,  4], dtype=int32)]
        """
        output = self.encode(x)
        return output

##############################################################################
# GRU class
##############################################################################

class Leak_GRU:
    """
    Constructs a GRU-based encoder, with modifications for initial conditions
    and style of output.

    Notation:
        Any variable name containing 'W' is a weight matrix;
        any variable name containing 'b' is a bias.

        The first subscript in a variable name refers to the function it
        helps; the second subcript refers to the argument it acts on.

        u = update gate
        r = forget gate
        h = state update function

    Args:
        embed_dim: dimension of the inputs into the GRU
        latent_dim: hidden dimension output from the GRU

    The memory vector in the decoder is the dimension of the inputs.

    >>> tf.reset_default_graph()
    >>> rnn = GeoGRU(4, 6)
    """
    def __init__(self, state_dim, embed_dim):
        self.state_dim = state_dim  # Dimension of the hidden layer
        self.embed_dim = embed_dim  # Dimension of the input layer
        # Declare the TensorFlow variables
#        gv(self, 'state_update', 'state', embed_dim, state_dim)
        with tf.variable_scope('l1'):
            self.l1 = Connected(state_dim, embed_dim + state_dim)
        with tf.variable_scope('l2'):
            self.l2 = Connected(state_dim, state_dim)
        with tf.variable_scope('l3'):
            self.l3 = Dense(1, state_dim, 2)

        # Declare properties that class functions will assign values to
        self.h = None
        self.h_set = None

    def update_state(self, x, h):
        """
        Takes an input vector and state and returns an updated state according
        to GRU logic:
            u = sigma(W_{ux} x + W_{uh} h0 + b_u)
            r = sigma(W_{rx} x + W_{uh} h0 + b_r)
            h' = tanh(W_{hx} x + r*W_{hh} h + b_h)
            h1 = (1-u)*h' + u*h
        where * is the Hadamard product,
        normal multiplication is implied matrix multiplication.

        Args:
            x: Rank 2 tensor of size [batch_size]*[embed_dim];
                this is the word used to update the hidden state
            h: Rank 2 tensor of size [batch_size]*[latent_dim];
                this is the current hidden state
        Returns:
            A rank 2 tensor of size [batch_size]*[latent_dim];
            this is the updated hidden state

        >>> tf.reset_default_graph()
        >>> rnn = GeoGRU(4,6)
        >>> word = tf.random_uniform([10,6])
        >>> state0 = tf.random_uniform([10,4])
        >>> state1 = rnn.update_state(word,state0)
        >>> shape = tf.shape(state1)
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> sess.run(shape)
        array([10,  4], dtype=int32)
        """
#        new_state = lgf(self, 'state', x, h)
        inputs = tf.concat([x, h], axis=-1)
        new_state = self.l2(self.l1(inputs))

        return new_state

    def encode(self, x):
        """
        Iterates over word vectors to return a hidden state encoding.

        Args:
            x: A rank 3 tensor of size [batch_size]*[token_num]*[embed_dim]
        Returns:
            A rank 2 tensors of size [batch_size]*[latent_dim]
        Additional:
            Adds sequence of hidden states constructed while encoding as
            property of the gru class.

        >>> tf.reset_default_graph()
        >>> sent = tf.random_uniform([10,8,6])
        >>> rnn = GeoGRU(4,6)
        >>> encoding = rnn.encode(sent)
        >>> shapes = [tf.shape(encoding), tf.shape(rnn.h_set)]
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> sess.run(shapes)
        [array([10,  4], dtype=int32), array([10,  8,  4], dtype=int32)]
        """
        init_state = (tf.ones(shape=[tf.shape(x)[0], self.state_dim]) /
                      tf.sqrt(tf.cast(self.state_dim, tf.float32)))
        x_t = tf.transpose(x, perm=[1, 0, 2])
        h_set_t = tf.scan(lambda h, x: self.update_state(x, h),
                          x_t, init_state)
        h_set = tf.transpose(h_set_t, perm=[1, 0, 2])

        # The sequence of hidden states constructed while encoding
        self.h_set = tf.map_fn(lambda x: tf.sigmoid(self.l3(x)), h_set)
        # The final encoding of the sequence of embedding vectors
        self.h = h_set[:, -1]

        return self.h

    def __call__(self, x):
        """
        Simply calls gru.encode(x).

        Args:
            x: A rank 3 tensor of size [batch_size]*[token_num]*[embed_dim]
        Returns:
            A rank 2 tensors of size [batch_size]*[latent_dim]

        >>> tf.reset_default_graph()
        >>> sent = tf.random_uniform([10,8,6])
        >>> rnn = GeoGRU(4,6)
        >>> encoding = rnn(sent)
        >>> shapes = [tf.shape(encoding), tf.shape(rnn.h_set)]
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> sess.run(shapes)
        [array([10,  4], dtype=int32), array([10,  8,  4], dtype=int32)]
        """
        output = self.encode(x)
        return output

##############################################################################
# Dense Class
##############################################################################

class Dense:
    def __init__(self, hid_dim, vec_dim, neuron_no):
        self.W_x = tf.get_variable(name='dense.W_x',
                                   shape=[neuron_no, hid_dim, vec_dim],
                                   initializer=tf.glorot_uniform_initializer())
        self.W = tf.get_variable(name='dense.W',
                               shape=[neuron_no],
                               initializer = tf.glorot_uniform_initializer())
        self.b = tf.get_variable(name="dense.b",
                                 shape=[neuron_no, hid_dim],
                                 initializer=tf.glorot_uniform_initializer())
        self.h = None
    def __call__(self, x):
        self.h = tf.einsum('i,kij->kj',
                           self.W,
                           tf.tanh(tf.einsum('ijk,lk->lij', self.W_x, x) +
                                   self.b))
        return tf.squeeze(self.h)


##############################################################################
# Gauge BC Class
##############################################################################

class Gauge_dense:
    def __init__(self, word_dim, latent_dim, neuron_no):
        self.W_x = tf.get_variable(name='gauge_dense.W_x',
                                   shape=[latent_dim, neuron_no,
                                          latent_dim + 1, word_dim],
                                   initializer=tf.glorot_uniform_initializer())
        self.W = tf.get_variable(name='gauge_dense.W',
                               shape=[neuron_no],
                               initializer = tf.glorot_uniform_initializer())
        self.b = tf.get_variable(name="gauge_dense.b",
                                 shape=[neuron_no, latent_dim + 1, word_dim],
                                 initializer=tf.glorot_uniform_initializer())
    def __call__(self, x):
        x_mul = tf.einsum('ij,jklm->iklm', x, self.W_x)
        nonlinear_layer = tf.tanh(x_mul + self.b)
        linear_layer = tf.einsum('j,ijkl->ikl', self.W, nonlinear_layer)
        return linear_layer


##############################################################################
# Boundary Discriminator Dense Class
##############################################################################

class Disc_dense:
    def __init__(self, vec_dim, neuron_no):
        self.W_x = tf.get_variable(name='dense.W_x',
                                   shape=[vec_dim, neuron_no],
                                   initializer=tf.glorot_uniform_initializer())
        self.W = tf.get_variable(name='dense.W',
                                 shape=[neuron_no],
                                 initializer = tf.glorot_uniform_initializer())
        self.b = tf.get_variable(name="dense.b",
                                 shape=[neuron_no],
                                 initializer=tf.glorot_uniform_initializer())
        with tf.variable_scope('batch_l1'):
            self.l1 = Connected(vec_dim, vec_dim)
        with tf.variable_scope('batch_l2'):
            self.l2 = Connected(vec_dim, vec_dim)
        with tf.variable_scope('batch_l3'):
            self.l3 = Connected(neuron_no, vec_dim)
    def __call__(self, x):
#        x_mul = tf.matmul(x, self.W_x)
#        nonlinear_layer = tf.tanh(x_mul + self.b)
        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        linear_layer = tf.einsum('i,ki->k', self.W, l3_out)
        return linear_layer


##############################################################################
# Boundary Discriminator Dense Class
##############################################################################

class Recon_dense:
    def __init__(self, state_dim, vec_dim, neuron_no):
        with tf.variable_scope("bc_l1"):
            self.l1 = Connected(state_dim, vec_dim)
        with tf.variable_scope("bc_l2"):
            self.l2 = Connected(2 * state_dim, state_dim)
        with tf.variable_scope("bc_l3"):
            self.l3 = Connected(2 * state_dim, 4 * state_dim)
#        with tf.variable_scope("bc_l4"):
#            self.l4 = Connected(2 * state_dim, 2 * state_dim)
#        with tf.variable_scope("bc_l5"):
#            self.l5 = Dense(2 * state_dim, 2 * state_dim, 30)
#            self.b = tf.get_variable(name='bias',
#                                     shape=[2 * state_dim],
#                                     initializer = tf.glorot_uniform_initializer())

    def __call__(self, x, y):
#        inputs = tf.concat([x,y], axis=-1)
        l1_out = self.l1(y)
        l2_out = self.l2(l1_out)
        inputs = tf.concat([x, l2_out], axis=-1)
        l3_out = self.l3(inputs)
#        l4_out = self.l4(l3_out)
#        l5_out = self.l5(l4_out)
#        return tf.tanh(l5_out + self.b)
        return l3_out


##############################################################################
# Leaked Discriminator Dense Class
##############################################################################

class Leak:
    def __init__(self, vec_dim):
        self.score = Leak_GRU(vec_dim, vec_dim)

    def __call__(self, x):
        self.score(x)
        scores = tf.squeeze(self.score.h_set)
        return scores

##############################################################################
# Fully connected Class
##############################################################################

class Connected:
    def __init__(self, hid_dim, vec_dim, act=tf.tanh):
        self.W_x = tf.get_variable(name='connected.W_x',
                                   shape=[vec_dim, hid_dim],
                                   initializer=tf.glorot_uniform_initializer())
        self.b = tf.get_variable(name="connected.b",
                                 shape=[hid_dim],
                                 initializer=tf.glorot_uniform_initializer())
        self.h = None
        self.act = act
    def __call__(self, x):
        self.h = self.act(tf.matmul(x, self.W_x) + self.b)
        return self.h


##############################################################################
# Word Classifier Class
##############################################################################

class Word_Class:
    def __init__(self, embed_dim, vocab_size):
        with tf.variable_scope("word_connected_layer"):
            self.classifier = Dense(vocab_size, embed_dim, 10)
#            self.classifier = Connected(vocab_size, embed_dim)
#            self.W = tf.get_variable(name='word_class_w',
#                                     shape=[vocab_size, vocab_size],
#                                     initializer=tf.glorot_uniform_initializer())
    def __call__(self, x):
        x_t = tf.transpose(x, perm=[1,0,2])
        pdf_t = tf.map_fn(lambda y: self.classifier(y), x_t)
        pdf = tf.transpose(pdf_t, perm=[1,0,2])
        return tf.nn.softmax(pdf)
#        return tf.nn.softmax(tf.einsum('ijk,kl->ijl', pdf, self.W))


##############################################################################
# Comparator Class
##############################################################################

class Comparator:
    """
    Discriminator alternative to MSE. Advantage: produces NLL-like loss;
    disadvantage: must be trained in adversarial manner to work.
    """
    def __init__(self, dim, neuron_no):
        self.dim = dim
        with tf.variable_scope("comp_dense"):
            self.dense = Dense(1, 2 * dim, neuron_no)
    def __call__(self, x, y):
        z = tf.concat([x,y], axis=-1)
        shape = z.shape.as_list()
        z_r = tf.reshape(x, shape=[-1, 2 * self.dim])
        energy_r = self.dense(z_r)
        energy = tf.reshape(energy_r, shape=[shape[0], -1])
        sig = tf.sigmoid(energy)
        return sig


##############################################################################
# Gauge and Related Classes
##############################################################################

class Gauge_evolver:
    def __init__(self, state_dim, word_dim, latent_dim):
        with tf.variable_scope('nonlinearity'):
            self.nonlinearities = Nonlinearity(word_dim, 2 * state_dim)
        with tf.variable_scope('current'):
            self.currents = Current(word_dim, state_dim)
        with tf.variable_scope('nonlinear_current'):
            self.F = Dense(word_dim, 10 * word_dim + 3 *  word_dim + 2 * state_dim,
                           10) # +  2 * state_dim
        with tf.variable_scope('w_gate'):
            self.gate = Gauge_gate(word_dim, 2 * word_dim + 2 * state_dim)
        self.state_dim = state_dim
        self.word_dim = word_dim
        self.latent_dim = latent_dim

    def __call__(self, w0, h, hbar, z):
        nonlinearities = self.nonlinearities(w0, z)
        currents = self.currents(w0, h, hbar)
        updates = tf.concat([currents, nonlinearities, z], axis=1)

        u = self.F(updates)

        inputs = tf.concat([u, w0, h, hbar], axis=-1)
        g = self.gate(inputs)
        w = (1 - g) * w0 + g * u

        return w

class Nonlinearity:
    def __init__(self, word_dim, latent_dim):
        m = 3
        initializer = tf.glorot_uniform_initializer()
        self.W_w = tf.get_variable(name='nonlinearity.W_w',
                                   shape=[word_dim,
                                          m * word_dim],
                                   initializer=initializer)
        self.W_z = tf.get_variable(name='nonlinearity.W_z',
                                   shape=[latent_dim,
                                          m * word_dim],
                                   initializer=initializer)
        self.b = tf.get_variable(name='nonlinearity.b',
                                 shape=[m * word_dim],
                                 initializer=initializer)

    def __call__(self, w, z):
        w_mul = tf.matmul(w, self.W_w)
        z_mul = tf.matmul(z, self.W_z)
        return tf.tanh(w_mul + z_mul + self.b)

class Current:
    def __init__(self, word_dim, state_dim):
        m = 10
        initializer = tf.glorot_uniform_initializer()
        self.W_w = tf.get_variable(name='current.W_w',
                                   shape=[word_dim, m * word_dim],
                                   initializer=initializer)
        self.W_h = tf.get_variable(name='current.W_h',
                                   shape=[2 * state_dim,
                                          m * word_dim],
                                   initializer=initializer)
        self.W_curr = tf.get_variable(name='current.W_curr',
                                      shape=[m * word_dim,
                                             m * word_dim],
                                      initializer=initializer)
        self.b_h = tf.get_variable(name='current.b_h',
                                   shape=[m * word_dim],
                                   initializer=initializer)
        self.b = tf.get_variable(name='current.b',
                                 shape=[m * word_dim],
                                 initializer=initializer)

    def __call__(self, w, h, hbar):
        state = tf.concat([h,hbar], axis=-1)
        curr = tf.tanh(tf.matmul(state, self.W_h) + self.b_h)
        curr_mul = tf.matmul(curr, self.W_curr)
        w_mul = tf.matmul(w, self.W_w)
        currents = tf.tanh(w_mul + curr_mul + self.b)
        return currents

class Gauge_gate:
    def __init__(self, hid_dim, vec_dim):
        self.W_x = tf.get_variable(name='connected.W_x',
                                   shape=[vec_dim, hid_dim],
                                   initializer=tf.glorot_uniform_initializer())
        self.b = tf.get_variable(name="connected.b",
                                 shape=[hid_dim],
                                 initializer=tf.glorot_uniform_initializer())
    def __call__(self, x):
        x_mul = tf.matmul(x, self.W_x)
        return tf.sigmoid(x_mul + self.b)


#class Gauge_evolver:
#    def __init__(self, state_dim, word_dim, latent_dim):
#        with tf.variable_scope('nonlinearity'):
#            self.nonlinearities = Nonlinearity(word_dim, latent_dim)
#        with tf.variable_scope('current'):
#            self.currents = Current(word_dim, state_dim)
#        with tf.variable_scope('nonlinear_current'):
#            self.F = Dense(word_dim, 2 * word_dim, 10)
#        with tf.variable_scope('w_gate'):
#            self.w_gate = Gauge_gate(word_dim, 2 * word_dim, latent_dim)
#        self.state_dim = state_dim
#        self.word_dim = word_dim
#        self.latent_dim = latent_dim
#
#    def __call__(self, w0, h, hbar, z):
#        nonlinearities = self.nonlinearities(w0, z)
#        currents = self.currents(w0, h, hbar)
#        updates = tf.concat([nonlinearities, currents], axis=2)
#        updates_t = tf.transpose(updates, perm=[1, 0, 2])
#
#        u_t = tf.map_fn(lambda x: self.F(x), updates_t)
#        u = tf.transpose(u_t, perm=[1, 0, 2])
#
#        inputs = tf.concat([u, w0], axis=-1)
#        w = (1 - self.w_gate(inputs)) * w0 + self.w_gate(inputs) * u
#
#        return w
#
#class Nonlinearity:
#    def __init__(self, word_dim, latent_dim):
#        initializer = tf.glorot_uniform_initializer()
#        self.W_w = tf.get_variable(name='nonlinearity.W_w',
#                                   shape=[(latent_dim + 1),
#                                          (latent_dim + 1),
#                                          word_dim,
#                                          word_dim],
#                                   initializer=initializer)
#        self.W_z = tf.get_variable(name='nonlinearity.W_z',
#                                   shape=[latent_dim + 1, latent_dim,
#                                          word_dim],
#                                   initializer=initializer)
#        self.b = tf.get_variable(name='nonlinearity.b',
#                                 shape=[latent_dim + 1, word_dim],
#                                 initializer=initializer)
#
#    def __call__(self, w, z):
#        w_mul = tf.einsum('ijl,jklm->ikm', w, self.W_w)
#        z_mul = tf.einsum('ij,kjl->ikl', z, self.W_z)
#        return tf.tanh(w_mul + z_mul + self.b)
#
#class Current:
#    def __init__(self, word_dim, state_dim):
#        initializer = tf.glorot_uniform_initializer()
#        self.W_w = tf.get_variable(name='current.W_w',
#                                   shape=[word_dim, word_dim],
#                                   initializer=initializer)
#        self.W_h = tf.get_variable(name='current.W_h',
#                                   shape=[2 * state_dim,
#                                          word_dim],
#                                   initializer=initializer)
#        self.W_curr = tf.get_variable(name='current.W_curr',
#                                      shape=[word_dim,
#                                             word_dim],
#                                      initializer=initializer)
#        self.b_h = tf.get_variable(name='current.b_h',
#                                   shape=[word_dim],
#                                   initializer=initializer)
#        self.b = tf.get_variable(name='current.b',
#                                 shape=[word_dim],
#                                 initializer=initializer)
#
#    def __call__(self, w, h, hbar):
#        state = tf.concat([h,hbar], axis=-1)
#        curr = tf.tanh(tf.matmul(state, self.W_h) + self.b_h)
#        curr_mul = tf.matmul(curr, self.W_curr)
#        w_mul = tf.matmul(w[:,0], self.W_w)
#        charge = tf.expand_dims(tf.tanh(w_mul + curr_mul + self.b), axis=1)
#        spatial_currents = tf.zeros_like(w)
#        currents = tf.concat([charge, spatial_currents[:,1:]], axis=1)
#        return currents
#
#class Gauge_gate:
#    def __init__(self, hid_dim, vec_dim, latent_dim):
#        self.W_x = tf.get_variable(name='connected.W_x',
#                                   shape=[latent_dim + 1, vec_dim, hid_dim],
#                                   initializer=tf.glorot_uniform_initializer())
#        self.b = tf.get_variable(name="connected.b",
#                                 shape=[latent_dim + 1, hid_dim],
#                                 initializer=tf.glorot_uniform_initializer())
#    def __call__(self, x):
#        x_mul = tf.einsum('ijk,jkl->ijl', x, self.W_x)
#        return tf.sigmoid(x_mul + self.b)
