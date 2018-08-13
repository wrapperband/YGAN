"""
@author: Sean A. Cantrell and Robin N. Tully
"""
import tensorflow as tf
from modules import basic_nets as bn
#from modules.GaugeNN import GaugeNN
from modules.simple_gauge import Gauge
from modules.batch_dis import Batch_Dis

from utilities import data_prep as dp

import numpy as np
from tqdm import tqdm
import codecs
import csv


class Benchmark:
    def __init__(self, state_dim, word_dim, latent_dim):
        self.state_dim = state_dim
        self.word_dim = word_dim
        self.latent_dim = latent_dim
        with codecs.open('./data/image_coco.txt','r',encoding='utf-8',errors='replace') as file:
            DATA = [row[0] for row in csv.reader(file)]
        freq_threshold = 5
        (self.train_set,
         self.test_set,
         self.vocab,
         self.word_id,
         self.id_word) = dp.convert_data(DATA, freq_threshold)
        vocab_size = len(self.vocab)
        self.test_set = padding_scalar(self.test_set)
        self.train_set = padding_scalar(self.train_set)

        tf.reset_default_graph()
        with tf.variable_scope('generator'):
            self.gen = Gauge(self.state_dim, self.word_dim, self.latent_dim)
        with tf.variable_scope('word_classifier'):
            self.word_class = bn.Word_Class(self.word_dim, vocab_size)
        initializer = tf.orthogonal_initializer()
        with tf.variable_scope('embeddings'):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                    shape=[vocab_size, self.word_dim],
                                    initializer=initializer)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, 'saved/caption_serve.ckpt')
    
    def autoencoder(self, real_inds):
        real_inds = tf.convert_to_tensor(real_inds)
        real = tf.nn.embedding_lookup(self.embedding_matrix, real_inds)
        real_words = vec2text(real, self.word_class, self.id_word, self.sess)
        synth, _, _ = self.gen.autoencode(real)
        synth_words = vec2text(synth, self.word_class, self.id_word, self.sess)
        synth_pdf = self.word_class(synth)
        acc = self.sess.run(word_class_acc(real_inds, synth_pdf))
        print("Real sentence:\n")
        for i, sent in enumerate(real_words):
            print(str(i) + "   " + sent)
        print("\nAutoencoded sentence:\n")
        for i, sent in enumerate(synth_words):
            print(str(i) + "   " + sent)
        print("\nAccuracy:", acc)
        

# Convert vectors to words
def vec2text(sents, mapper, id_word, sess):
    pdf = mapper(sents)
    vocab_size = pdf.shape.as_list()[-1]
    hots = tf.squeeze(tf.one_hot(tf.nn.top_k(pdf).indices, vocab_size))
    word_inds = sess.run(tf.argmax(hots, axis=2))
    text = [' '.join([id_word[ind] for ind in row]) for row in word_inds]
    return text

# Input padding
def padding_scalar(df, min_length = 1):
    pad = [0]
    df = np.array([list(elem) + pad for elem in df])
    lens = np.array([len(batch) for batch in df])
    lens_max = max(max(lens), min_length)
    df = np.array([list(elem) + pad*(lens_max - lens[i])
                  for i,elem in enumerate(df)])
    return df

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