"""
GaugeGAN support functions
@author: Sean A. Cantrell
"""
import tensorflow as tf
import numpy as np

##############################################################################
# Support functions
##############################################################################

# Loss function
def cross_entropy(pred):
    delta = 1e-12
    cross_ent = -tf.reduce_sum(tf.log(pred + delta))
    return cross_ent

# mse loss
def mse(real, pred):
    diff = tf.sqrt(tf.reduce_sum((real - pred)**2, axis=-1))
    means = tf.reduce_mean(diff)
    return means

# NLL-MSE loss
def msll(real, pred):
    delta = 1e-5
    diff = tf.sqrt(tf.reduce_sum((pred - real)**2, axis=-1))
    sig_diff = tf.exp(diff)
    nll = tf.reduce_mean(tf.log(tf.maximum(sig_diff - 1, delta)))
    return nll

# MSE acc
def ms_acc(real_inds, pred, embeddings):
    real = tf.nn.embedding_lookup(embeddings, real_inds)
    diff = tf.sqrt(tf.reduce_sum((pred - real)**2, axis=-1))
    norm = tf.sqrt(tf.reduce_sum(pred**2, axis=-1))

    max_len = real_inds.shape.as_list()[1]
    bool_mask = tf.equal(real_inds, 0)
    sent_lengths = tf.argmax(tf.cast(bool_mask, tf.int32), axis=-1)
    seq_mask = tf.cast(tf.sequence_mask(sent_lengths + 1, max_len),
                       tf.float32)    

    sig_diff = (tf.reduce_sum(seq_mask * (1. - diff / norm), axis=-1) /
                (tf.cast(sent_lengths, tf.float32) + 1))
    return tf.reduce_mean(sig_diff)

def ms_state_acc(real_inds, real_states, pred_states):
    diff = tf.sqrt(tf.reduce_sum((real_states - pred_states)**2, axis=-1))
    norm = tf.sqrt(tf.reduce_sum(pred_states**2, axis=-1))

    max_len = real_inds.shape.as_list()[1]
    bool_mask = tf.equal(real_inds, 0)
    sent_lengths = tf.argmax(tf.cast(bool_mask, tf.int32), axis=-1)
    seq_mask = tf.cast(tf.sequence_mask(sent_lengths + 1, max_len),
                       tf.float32) 

    sig_diff = (tf.reduce_sum(seq_mask * (1 - diff / norm), axis=-1) /
                (tf.cast(sent_lengths, tf.float32) + 1))
    return tf.reduce_mean(sig_diff)

# Word classifier-related loss functions ######################################
def adversarial_loss_func(real_states, synth_states,
                          leak):
    delta = 1e-12

    q_real = leak(real_states)
    q_synth = leak(synth_states)

    cross_ent = (-tf.log(tf.maximum(q_real * (1 - q_synth), delta)) +
                 tf.log(tf.maximum(q_real + q_synth - 2 * q_real * q_synth, delta)))
                     
    return tf.reduce_mean(cross_ent)

def auto_loss_func(real_inds, pdf,
                    real, pred,
                    real_states, pred_states):
    delta = 1e-12
    vocab_size = pdf.shape.as_list()[-1]
    real_hots = tf.one_hot(real_inds, vocab_size)
    pdf = tf.reduce_sum(real_hots * pdf, axis=-1)

    diff = tf.sqrt(tf.reduce_sum((pred - real)**2, axis=-1))
    diff_states = tf.sqrt(tf.reduce_sum((pred_states - real_states)**2, axis=-1))
    
    sig_diff = tf.exp(-diff)
    sig_diff_states = tf.exp(-diff_states)

    p = sig_diff * pdf
    
    cross_ent = (-tf.log(tf.maximum(p, delta)) +
                 tf.log(tf.maximum(1 - p, delta)))
                     
    return tf.reduce_mean(cross_ent)

def state_loss_func(real_states, pred_states):
    delta = 1e-12

    diff_states = tf.sqrt(tf.reduce_sum((pred_states - real_states)**2, axis=-1))
    p = tf.exp(-diff_states)
    
    cross_ent = (-tf.log(tf.maximum(p, delta)) +
                 tf.log(tf.maximum(1 - p, delta)))

    return tf.reduce_mean(cross_ent)


def word_class_msll(real_inds, pdf, real, pred):
    delta = 1e-12
    vocab_size = pdf.shape.as_list()[-1]
    real_hots = tf.one_hot(real_inds, vocab_size)
    pdf = tf.reduce_sum(real_hots * pdf, axis=-1)

    diff = tf.sqrt(tf.reduce_sum((pred - real)**2, axis=-1))
    sig_diff = tf.exp(-diff)

    p = sig_diff * pdf
    cross_ent_1 = tf.log(tf.maximum(1 - p, delta))
    cross_ent_2 =  -tf.log(tf.maximum(p, delta))
    cross_ent = (cross_ent_1 + cross_ent_2)
    return tf.reduce_mean(cross_ent)

def word_class_msll_adv(real_inds, pdf,
                        real, pred,
                        real_states, synth_states,
                        leak, gen=True):
    delta = 1e-12
    vocab_size = pdf.shape.as_list()[-1]
    real_hots = tf.one_hot(real_inds, vocab_size)
    pdf = tf.reduce_sum(real_hots * pdf, axis=-1)

    diff = tf.sqrt(tf.reduce_sum((pred - real)**2, axis=-1))
    sig_diff = tf.exp(-diff)

    q_real = leak(real_states)
    q_synth = leak(synth_states)

    p_real = sig_diff * pdf * q_real
    p_synth = sig_diff * pdf #* q_synth
    
    if gen:
        cross_ent =  ( - tf.log(tf.maximum(p_synth, delta)) +
                      tf.log(tf.maximum(1 - p_synth, delta)))
    else:
        cross_ent = (-tf.log(tf.maximum(p_real, delta)) -
                     tf.log(tf.maximum(1 - p_synth, delta)) +
                     tf.log(tf.maximum(p_real + p_synth - 2 * p_real * p_synth, delta)))
                     
    return tf.reduce_mean(cross_ent)
#    cross_ent = (cross_ent_2 - (1 - 2 * float(gen)) * cross_ent_1)
#    return tf.reduce_mean(cross_ent)


def word_class_msll_mask(real_inds, pdf, real, pred, meanQ=False):
    delta = 1e-12
    vocab_size = pdf.shape.as_list()[-1]
    real_hots = tf.one_hot(real_inds, vocab_size)
    pred_inds = tf.argmax(pdf, axis=-1)
    pred_hots = tf.one_hot(pred_inds, vocab_size)
    pdf = tf.reduce_sum(real_hots * pdf, axis=-1)

    diff = tf.sqrt(tf.reduce_sum((pred - real)**2, axis=-1))
    sig_diff = tf.exp(-diff)

    p = sig_diff * pdf

    acc = tf.reduce_sum(real_hots * pred_hots, axis=-1)
    max_len = acc.shape.as_list()[1]
    bool_mask = tf.equal(acc, 0)
    sent_lengths = tf.argmax(tf.cast(bool_mask, tf.int32), axis=-1)
    seq_mask = tf.cast(tf.sequence_mask(sent_lengths + 1, max_len),
                       tf.float32)

    cross_ent_1 = tf.log(tf.maximum(1 - p, delta))
    cross_ent_2 =  -tf.log(tf.maximum(p, delta))
    cross_ent = seq_mask * (cross_ent_1 + cross_ent_2)
    if meanQ:
        return tf.reduce_mean(cross_ent)
    else:
        return tf.reduce_sum(cross_ent)

def word_class_loss(real_inds, pred, meanQ=False):
    delta = 1e-5
    vocab_size = pred.shape.as_list()[-1]
    real = tf.one_hot(real_inds, vocab_size)
    cross_ent_1 =  tf.reduce_sum(real *
                                 tf.log(tf.maximum(1 - pred, delta)), axis=-1)
    cross_ent_2 =  -tf.reduce_sum(real *
                                 tf.log(tf.maximum(pred, delta)), axis=-1)
    if meanQ:
        return tf.reduce_mean(cross_ent_1 + cross_ent_2)
    else:
        return tf.reduce_sum(cross_ent_1 + cross_ent_2)

def word_class_nll(real_inds, pred, meanQ=False):
    delta = 1e-5
    vocab_size = pred.shape.as_list()[-1]
    real = tf.one_hot(real_inds, vocab_size)
    cross_ent =  -tf.reduce_sum(real *
                                tf.log(tf.maximum(pred, delta)), axis=-1)
    if meanQ:
        return tf.reduce_mean(cross_ent)
    else:
        return tf.reduce_sum(cross_ent)

def word_class_acc(real_inds, pred):
    vocab_size = pred.shape.as_list()[-1]
    real_hots = tf.one_hot(real_inds, vocab_size)
    pred_hots = tf.squeeze(tf.one_hot(tf.nn.top_k(pred).indices, vocab_size))

    max_len = real_inds.shape.as_list()[1]
    bool_mask = tf.equal(real_inds, 0)
    sent_lengths = tf.argmax(tf.cast(bool_mask, tf.int32), axis=-1)
    seq_mask = tf.cast(tf.sequence_mask(sent_lengths + 1, max_len),
                       tf.float32)

    accs = tf.reduce_sum(real_hots * pred_hots, axis=-1)
    means = tf.reduce_sum(seq_mask * accs, axis=-1) / (tf.cast(sent_lengths, tf.float32) + 1)
    return tf.reduce_mean(means)

##############################################################################
def mask(real, pad_vec):
    max_len = real.shape.as_list()[1]
    bool_mask = tf.reduce_all(tf.equal(real, pad_vec), axis=-1)
    sent_lengths = tf.argmax(tf.cast(bool_mask, tf.int32), axis=-1)
    seq_mask = tf.cast(tf.sequence_mask(sent_lengths+1, max_len), tf.float32)
    return seq_mask

def mask_bar(real, pad_vec):
    max_len = real.shape.as_list()[1]
    bool_mask = tf.reduce_all(tf.equal(real, pad_vec), axis=-1)
    sent_lengths = tf.argmax(tf.cast(bool_mask, tf.int32), axis=-1)
    seq_mask_bar = tf.cast(tf.logical_not(tf.sequence_mask(max_len -
                                                           sent_lengths -
                                                           1,
                                                           max_len)),
                           tf.float32)
    return seq_mask_bar

def mask_states(real, pad_vec, states):
    shape = states.shape.as_list()
    state_dim = int(shape[-1]/2)
    seq_mask = mask(real, pad_vec)
    seq_mask_bar = mask_bar(real, pad_vec)
    states_forw = tf.einsum('ij,ijk->ijk', seq_mask, states[:,:,:state_dim])
    states_back = tf.einsum('ij,ijk->ijk', seq_mask_bar, states[:,:,state_dim:])
    masked_states = tf.concat([states_forw, states_back], axis=-1)
    return masked_states

# Discriminator accuracy function
def dis_accuracy(real, pred):
    depth = tf.shape(pred)[1]
    pred_hot = tf.squeeze(tf.one_hot(tf.nn.top_k(pred).indices, depth))
    return tf.reduce_sum(real*pred_hot)

# Generator accuracy function
def gen_accuracy(real, pred, embedding_matrix):
    return 'Not today, son'

# Input padding
def padding_scalar(df, min_length = 1):
    pad = [0]
    df = np.array([list(elem) + pad for elem in df])
    lens = np.array([len(batch) for batch in df])
    lens_max = max(max(lens), min_length)
    df = np.array([list(elem) + pad*(lens_max - lens[i])
                  for i,elem in enumerate(df)])
    return df

# Convert vectors to words
def vec2text(sents, mapper, id_word, sess):
    pdf = mapper(sents)
    vocab_size = pdf.shape.as_list()[-1]
    hots = tf.squeeze(tf.one_hot(tf.nn.top_k(pdf).indices, vocab_size))
    word_inds = sess.run(tf.argmax(hots, axis=2))
    text = [[id_word[ind] for ind in row] for row in word_inds]
    return text


# Swap words within a sentence - ensure grammatical structure
def shuffle_sentences(sents):
    sents_shuffle = tf.map_fn(lambda x: shuffle_nonzero(x), sents)
    return sents_shuffle

def shuffle_nonzero(sent):
    sent_end = tf.argmax(tf.cast(tf.equal(sent, 0), tf.int32))
    short_shuffle = tf.random_shuffle(sent[:sent_end])
    sent_shuffled = tf.concat([short_shuffle, sent[sent_end:]], axis=0)
    return sent_shuffled


def batch_data(data, batch_size, embeddings):
    capacity = len(data)
    real_inds = tf.train.shuffle_batch(data,
                                       batch_size,
                                       capacity=capacity,
                                       min_after_dequeue=0,
                                       enqueue_many=True,
                                       allow_smaller_final_batch=False)
    fake_inds = shuffle_sentences(real_inds)
    collapse_inds = collapse_batch(real_inds)
    real = tf.nn.embedding_lookup(embeddings, real_inds)
    fake = tf.nn.embedding_lookup(embeddings, fake_inds)
    collapse = tf.nn.embedding_lookup(embeddings, collapse_inds)
    return real, fake, collapse, real_inds

def autoencode(real, gen):
    # Prep for operations
    length = real.shape.as_list()[1]

    # Encode the inputs
    encoding = get_states(real, gen)

    # Reconstruct latent variable
    z = gen.latent_recon(encoding[:,-1])

    # Generate synthetic sentence
    synth, h, hbar = gen.decode(z, length)
    states = tf.concat([h, hbar], axis=-1)
    return synth, states, encoding

def get_states(sent, gen):
    gen.state_forw(sent)
    states_forw = gen.state_forw.h_set
    sent_back = tf.reverse(sent, axis=[1])
    gen.state_back(sent_back)
    states_back = gen.state_back.h_set
    states = tf.concat([states_forw, states_back], axis=-1)
    return states

def collapse_batch(indices):
    shape = indices.shape.as_list()
    batch_size = shape[0]
    length = shape[1]
    real_entries = np.random.randint(low=0,
                                     high=int(np.floor(.7 * batch_size)))
    fake_entries = np.random.randint(low=real_entries+1,
                                     high=batch_size-1)
    real_bois = indices[:real_entries]
    fake_bois = tf.reshape(tf.tile(indices[fake_entries],
                                   [batch_size - real_entries]),
                           shape=[-1, length])
    collapse = tf.concat([real_bois, fake_bois], axis=0)
    output = tf.random_shuffle(collapse)
    return output

