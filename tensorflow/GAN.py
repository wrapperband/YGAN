"""
Meta-TextGAN class for training
@author: Sean A. Cantrell & Robin N. Tully

Maybe make the construction of loss functions a function, and train just calls
it.
Make the loss construction function out of simpler functions
"""
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

from modules import basic_nets as bn
from modules.simple_gauge import Gauge
from modules.batch_dis import Batch_Dis

from utilities import data_prep as dp
from utilities.support import *

from tqdm import tqdm
import json
import csv
import time


class GAN:
    def __init__(self, state_dim, word_dim, latent_dim):
        tf.reset_default_graph()
        self.word_dim = word_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Instantiate the generator
        with tf.variable_scope('generator'):
            self.gen = Gauge(state_dim, word_dim, latent_dim)

        # Instantiate the boundary discriminator
        with tf.variable_scope('discriminator'):
            self.disc = bn.Disc_dense(2 * state_dim, 2)

        # Instantiate the leak discriminator
        with tf.variable_scope('leak'):
            self.leak = bn.Leak(2 * state_dim)

        # Declare additional attributes
        self.embedding_matrix = None
        self.vocab = None
        self.word_id = None
        self.id_word = None

    def train(self, data,
              batch_size,
              epochs,
              phase=1):
        """
        Trains the GaugeGAN and discriminator using a modified TextGAN.
        Saves model parameters in ./model_parameters
        Stores model parameters in memory as attributes of self.gen for
        additional inference if desired (e.g. in ipython/jupyter)
        Args:
            data = single list of data - zipped text:category pairs.
                    Must be pre-shuffled. Train, validation, and test sets
                    will be constructed for this single set
            batch_size = Int, batch size to train D and G
            epochs = Int, epochs over which to train D and G
            iterations = Int, number of times to iteratively perform D + G
                        training
        """
        freq_threshold = 1
        (train_set,
         test_set,
         self.vocab,
         self.word_id,
         self.id_word) = dp.convert_data(data, freq_threshold)
        vocab_size = len(self.vocab)
        initializer = tf.orthogonal_initializer()
        with tf.variable_scope('embeddings'):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                    shape=[vocab_size, self.word_dim],
                                    initializer=initializer)

        ######################################################
        # Creating the training, validation, and hold out sets
        # for discriminator training
        ######################################################
        print("Creating training, validation, and testing sets")
        
        batch_no_train = int(np.ceil(len(train_set)/batch_size))
        batch_no_val = int(np.ceil(len(test_set)/batch_size))

        # Padding
        val_set = [padding_scalar(test_set)]
        train_set = [padding_scalar(train_set)]

        longest_train = len(train_set[0][0])
        longest_val = len(val_set[0][0])
        longest = max([longest_train, longest_val])
        if longest_val < longest:
            val_set = [padding_scalar(val_set[0], longest)]
        if longest_train < longest:
            train_set = [padding_scalar(train_set[0], longest)]

###############################################################################
        # Instantiate the batch discriminator
        with tf.variable_scope('batch_discriminator'):
            self.batch_disc = Batch_Dis(2 * self.state_dim, batch_size,
                                        128, 64)
        # Instantiate the word classifier
        with tf.variable_scope('word_classifier'):
            self.word_class = bn.Word_Class(self.word_dim, vocab_size)

        ###################################################
        # Setup the queue and runner for real and fake data
        ###################################################
        print("Setting up the training, validation, and test queues")
        (real_words_train,
         fake_words_train,
         collapse_words_train,
         real_inds_train) = batch_data(train_set, batch_size, self.embedding_matrix)
        real_pdf_train = self.word_class(real_words_train)
        real_states_train = self.gen.get_and_concat_states(real_words_train)

        (real_words_val,
         fake_words_val,
         collapse_words_val,
         real_inds_val) = batch_data(val_set, batch_size, self.embedding_matrix)
        real_pdf_val = self.word_class(real_words_val)
        real_states_val = self.gen.get_and_concat_states(real_words_val)


        ###################################################
        # Generate synthetic sentences and states
        ###################################################
        # Novel synthetic sentences
        z = tf.random_uniform(shape=[batch_size, self.latent_dim])
        z0 = tf.zeros_like(z)
        synth_bc = real_states_train[:,-1] + self.gen.state_recon(real_states_train[:,-1], z) / np.sqrt(2 * self.state_dim)
        synth_bc_auto = real_states_train[:,-1] + self.gen.state_recon(real_states_train[:,-1], z0) / np.sqrt(2 * self.state_dim)
        synth, _, _ = self.gen.decode(synth_bc, longest)
        synth_states = self.gen.get_and_concat_states(synth)


        # Schedule sentences
        skip_no = 10
        sched_words_train = []
        sched_states_train = []
        sched_pdf_train = []
        shape = real_words_train.shape.as_list()
        noise_dir = tf.random_uniform(shape=shape, minval=-1., maxval=1.)
        noise_dir = tf.nn.l2_normalize(noise_dir, axis=-1)
        noise = (1 / 70.) * tf.random_uniform(shape=[1]) * noise_dir
        for n in range(skip_no):
            sched_temp, forw_temp, back_temp = self.gen.schedule(real_words_train +
                                                                 noise,
                                                                 n + 1)
            states_temp = self.gen.get_and_concat_states(sched_temp)
            sched_pdf_temp = self.word_class(sched_temp)
            sched_words_train.append(sched_temp)
            sched_states_train.append(states_temp)
            sched_pdf_train.append(sched_pdf_temp)
            

        sched_words_val = []
        sched_pdf_val = []
        sched_states_val = []
        for n in range(skip_no):
            sched_temp, forw_temp, back_temp = self.gen.schedule(real_words_val, n + 1)
            states_temp = self.gen.get_and_concat_states(sched_temp)
            sched_pdf_temp = self.word_class(sched_temp)
            sched_words_val.append(sched_temp)
            sched_pdf_val.append(sched_pdf_temp)
            sched_states_val.append(states_temp)

        # Autoencoded sentences and states
        (auto_words_train,
         _,
         _) = self.gen.autoencode(real_words_train)
        auto_pdf_train = self.word_class(auto_words_train)
        auto_states_train = self.gen.get_and_concat_states(auto_words_train)

        (auto_words_val,
         _,
         _) = self.gen.autoencode(real_words_val)
        auto_pdf_val = self.word_class(auto_words_val)
        auto_states_val = self.gen.get_and_concat_states(auto_words_val)

        # Additional states
        # Training
        fake_states_train = tf.concat(self.gen.get_states(fake_words_train),
                                      axis=-1)[:,-1]
        collapse_states_train = tf.concat(self.gen.get_states(collapse_words_train),
                                          axis=-1)[:,-1]
        # Validation
        fake_states_val = tf.concat(self.gen.get_states(fake_words_val),
                                    axis=-1)[:,-1]
        collapse_states_val = tf.concat(self.gen.get_states(collapse_words_val),
                                        axis=-1)[:,-1]

        #############################
        # Loss ops
        #############################
        print("Constructing loss ops")
        delta = 1e-6
        # Phase 1 training
        # Adversarial scheduler losses
        loss_sched_class = [auto_loss_func(real_inds_train, pdf,
                                           real_words_train, sched_words_train[i],
                                           real_states_train, sched_states_train[i]) for
                            i, pdf in enumerate(sched_pdf_train)]
        loss_sched_disc = [adversarial_loss_func(real_states_train, state,
                                                 self.leak) for
                           state in sched_states_train]
        loss_sched_gen = [adversarial_loss_func(state, real_states_train,
                                                self.leak) for
                          state in sched_states_train]

        # Mapper losses
        loss_mapper = word_class_loss(real_inds_train, real_pdf_train, meanQ=True)

        # Phase 2 training
        # Adversarial autoencoder losses
        loss_auto_class = auto_loss_func(real_inds_train, auto_pdf_train,
                                         real_words_train, auto_words_train,
                                         real_states_train, auto_states_train)
        loss_auto_disc = adversarial_loss_func(real_states_train, auto_states_train,
                                               self.leak)
        loss_auto_gen = adversarial_loss_func(auto_states_train, real_states_train,
                                              self.leak)

        # Phase 3 training
        # Boundary discriminator losses
        c_fake = (self.disc(real_states_train[:,-1]) -
                  self.disc(fake_states_train))
        c_disc = (self.disc(real_states_train[:,-1]) -
                   self.disc(synth_bc))
        c_synth = (self.disc(synth_bc) -
                   self.disc(real_states_train[:,-1]))

        d_fake = tf.sigmoid(c_fake)
        d_synth = tf.sigmoid(c_synth)
        d_disc = tf.sigmoid(c_disc)

        noise = tf.random_uniform(shape=[batch_size, 2 * self.state_dim],
                                  minval=-1.,
                                  maxval=1.)
        noise = 1e-5 * tf.nn.l2_normalize(noise, axis=-1)
        synth_bc_exp = tf.expand_dims(synth_bc, axis=1)
        sym_dist = tf.sqrt(tf.reduce_sum((synth_bc - synth_bc_exp + noise)**2, axis=-1) /
                                               tf.reduce_sum(synth_bc**2, axis=-1))
#        states_sym_dist = tf.sqrt(tf.reduce_sum((real_states_train[:,-1] - synth_bc_exp + noise)**2, axis=-1) /
#                                  tf.reduce_sum(real_states_train[:,-1]**2, axis=-1))
        interaction = tf.map_fn(lambda i: tf.reduce_sum(1 / sym_dist[i,:i]) +
                                tf.reduce_sum(1 / sym_dist[i,i+1:]),
                                tf.range(0, batch_size),
                                dtype=tf.float32)
#        stabilizer = tf.sqrt(tf.reduce_sum((real_states_train[:,-1] - synth_bc)**2, axis=-1) /
#                             tf.reduce_sum(real_states_train[:,-1]**2, axis=-1))
#        dist_match = tf.map_fn(lambda i: tf.sqrt(tf.reduce_sum((states_sym_dist[i,:i] - sym_dist[i,:i])**2, axis=-1) +
#                                                 tf.reduce_sum((states_sym_dist[i,i+1:] - sym_dist[i,i+1:])**2, axis=-1))/ 
#                               tf.reduce_sum(states_sym_dist**2, axis=-1),
#                               tf.range(0, batch_size),
#                               dtype=tf.float32)
        bc_state_diff = tf.sqrt(tf.reduce_sum((real_states_train[:,-1] - synth_bc)**2, axis=-1))
        bc_norm = tf.sqrt(tf.reduce_sum(real_states_train[:,-1]**2, axis=-1))
        topic_stabilizer = tf.reduce_mean(bc_state_diff / bc_norm)

        loss_bc_disc = tf.reduce_mean(-tf.log(tf.maximum(d_disc, delta)))
        loss_bc_fake = tf.reduce_mean(-tf.log(tf.maximum(d_fake, delta)))
        loss_bc_gen = tf.reduce_mean(-tf.log(tf.maximum(d_synth, delta)) +
                                     topic_stabilizer**2 + 1e-5 * interaction)


        bc_state_diff_auto = tf.sqrt(tf.reduce_sum((real_states_train[:,-1] - synth_bc_auto)**2, axis=-1))
        loss_bc_gen_constraint = tf.reduce_mean(bc_state_diff_auto / bc_norm)


        # Phase 4 training
        loss_synth_disc = adversarial_loss_func(real_states_train, synth_states,
                                                self.leak)
        loss_synth_gen = adversarial_loss_func(synth_states, real_states_train,
                                               self.leak)

        # Losses for training functions
        loss_phase_1_disc = loss_sched_disc
        loss_phase_1_gen = loss_sched_gen
        loss_phase_1_class = [loss + loss_mapper for loss in loss_sched_class]

        loss_phase_2_disc = loss_auto_disc
        loss_phase_2_gen = loss_auto_gen
        loss_phase_2_class = loss_auto_class + loss_mapper

        loss_phase_3_disc = (loss_bc_disc + loss_bc_fake)
        loss_phase_3_gen = loss_bc_gen + 10 * loss_bc_gen_constraint
        loss_phase_3_auto = loss_bc_gen_constraint

        loss_phase_4_disc = loss_synth_disc #+ loss_phase_3_disc
        loss_phase_4_bc_gen = loss_synth_gen + loss_bc_gen + loss_bc_gen_constraint
        loss_phase_4_gen = loss_synth_gen
        loss_phase_4_class = loss_phase_2_class

        #############################
        # Training ops
        #############################
        print("Constructing training ops")
        # Establish variable lists
        # Phase 2 training
        auto_vars = (tf.trainable_variables('embeddings') +
                     tf.trainable_variables('generator'))
        word_class_vars = tf.trainable_variables('word_classifier')

        phase_1_vars_disc = tf.trainable_variables('leak')
        phase_1_vars_gen = (tf.trainable_variables('generator/invert_word') +
                            tf.trainable_variables('generator/gen_word'))
        phase_1_vars_class = auto_vars + word_class_vars

        phase_2_vars_disc = phase_1_vars_disc
        phase_2_vars_gen = phase_1_vars_gen
        phase_2_vars_class = phase_1_vars_class

        phase_3_vars_disc = (tf.trainable_variables('discriminator'))
        phase_3_vars_gen = tf.trainable_variables('generator/state_reconstruction')
        phase_3_vars_auto = tf.trainable_variables('generator/state_reconstruction')

        phase_4_vars_disc = phase_1_vars_disc + tf.trainable_variables('discriminator')
        phase_4_vars_gen = (tf.trainable_variables('generator/invert_word') +
                            tf.trainable_variables('generator/gen_word') +
                            tf.trainable_variables('generator/word_bc'))
#        phase_4_vars_gen = tf.trainable_variables('generator')
        phase_4_vars_class = tf.trainable_variables('generator')
                            

        # Create training ops
        opt = tf.train.AdamOptimizer(1e-3)
        opt_gen = tf.train.GradientDescentOptimizer(1e-3)
        train_1_disc = [opt.minimize(loss,
                                var_list=phase_1_vars_disc) for loss in loss_phase_1_disc]
        train_1_gen = [opt.minimize(loss,
                                    var_list=phase_1_vars_gen) for loss in loss_phase_1_gen]
        train_1_class = [opt.minimize(loss,
                                      var_list=phase_1_vars_class) for loss in loss_phase_1_class]

        train_2_disc = opt.minimize(loss_phase_2_disc,
                                    var_list=phase_2_vars_disc)
        train_2_gen = opt.minimize(loss_phase_2_gen,
                                   var_list=phase_2_vars_gen)
        train_2_class = opt.minimize(loss_phase_2_class,
                                     var_list=phase_2_vars_class)

        train_3_disc = opt.minimize(loss_phase_3_disc,
                                    var_list=phase_3_vars_disc)
        train_3_gen = opt.minimize(loss_phase_3_gen,
                                   var_list=phase_3_vars_gen)
        train_3_auto = opt.minimize(loss_phase_3_auto,
                                    var_list=phase_3_vars_auto)

        train_4_disc = opt.minimize(loss_phase_4_disc,
                                    var_list=phase_4_vars_disc)
        train_4_bc_gen = opt.minimize(loss_phase_4_bc_gen,
                                      var_list=phase_3_vars_gen)
        train_4_gen = opt.minimize(loss_phase_4_gen,
                                       var_list=phase_4_vars_gen)
        train_4_class = opt.minimize(loss_phase_4_class,
                                     var_list=phase_4_vars_class)

        #############################
        # Performance metrics ops
        #############################
        print("Constructing validation ops")
        # Phase 1 training
        # Scheduler metrics
        val_sched_class = [word_class_acc(real_inds_val, pdf) for pdf in sched_pdf_val]
        val_sched_train = [word_class_acc(real_inds_train, pdf) for pdf in sched_pdf_train]
        val_sched_comp = [ms_acc(real_inds_val, comp, self.embedding_matrix) for comp in sched_words_val]
        val_state_comp_sched = [ms_state_acc(real_inds_val, real_states_val, comp) for comp in sched_states_val]
        val_sched_nll = [word_class_nll(real_inds_val, pdf, meanQ=True) for pdf in sched_pdf_val]
        # Mapper metrics
        val_mapper = word_class_acc(real_inds_val, real_pdf_val)
        # Embedding norm
        val_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(self.embedding_matrix**2, axis=-1)))
        # Leak metrics
        leak_real = self.leak(real_states_val)
        leak_sched_synth = [self.leak(state) for state in sched_states_val]
        val_leak_sched_acc = [tf.reduce_mean(tf.round(leak_real) +
                              (1 - tf.round(val))) / 2 for val in leak_sched_synth]
        val_leak_sched_synth = [tf.reduce_mean(val) for val in leak_sched_synth]
        val_leak_real = tf.reduce_mean(leak_real)
        val_leak_sched_rel = [tf.reduce_mean(leak_real * (1 - val) /
                                             (leak_real + val - 2 * leak_real * val)) for
                                val in val_leak_sched_synth]

        # Phase 2 training
        # Autoencoder metrics
        epsilon = 1e-5
        val_auto_class = word_class_acc(real_inds_val, auto_pdf_val)
        val_auto_train = word_class_acc(real_inds_train, auto_pdf_train)
        val_auto_nll = word_class_nll(real_inds_val, auto_pdf_val, meanQ=True)
        val_auto_comp = ms_acc(real_inds_val, auto_words_val, self.embedding_matrix)
        val_state_auto_comp = ms_state_acc(real_inds_val, real_states_val, auto_states_val)
        val_auto_train = word_class_acc(real_inds_train, auto_pdf_train)
        val_leak_auto_acc = tf.reduce_mean(tf.round(self.leak(real_states_val)) +
                        (1 - tf.round(self.leak(auto_states_val)))) / 2
        val_leak_auto_synth = tf.reduce_mean(self.leak(auto_states_val))
        leak_auto_synth = self.leak(auto_states_val)
        val_leak_auto_rel = tf.reduce_mean(tf.maximum(leak_real * (1 - leak_auto_synth), epsilon) / 
                                      tf.maximum(leak_real + leak_auto_synth - 2 * leak_real * leak_auto_synth, epsilon))

        # Phase 3 training
        # BC metrics
        synth_bc_val = real_states_val[:,-1] + self.gen.state_recon(real_states_val[:,-1], z) / np.sqrt(2 * self.state_dim)
        synth_val, _, _ = self.gen.decode(synth_bc_val, longest)
        synth_states_val = self.gen.get_and_concat_states(synth_val)

        real_states_val_rand = tf.random_shuffle(real_states_val[:,-1])
        synth_bc_rev_val = real_states_val_rand + self.gen.state_recon(real_states_val_rand, z) / np.sqrt(2 * self.state_dim)
        val_synth_particle_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum((synth_bc_val - synth_bc_rev_val)**2, axis=-1)))
        val_real_particle_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum((real_states_val[:,-1] - real_states_val_rand)**2, axis=-1)))
        val_particle_dist_ratio = val_synth_particle_dist / val_real_particle_dist
        

        c_val_fake = (self.disc(real_states_val[:,-1]) -
                       self.disc(fake_states_val))
        c_val_batch_fake = (self.batch_disc(real_states_val[:,-1]) -
                            self.batch_disc(collapse_states_val))
        c_val_synth = (self.disc(synth_bc_val) -
                       self.disc(real_states_val[:,-1]))
        c_val_batch_synth = (self.batch_disc(synth_bc_val) -
                             self.batch_disc(real_states_val[:,-1]))

        d_val_fake = tf.sigmoid(c_val_fake)
        d_val_batch_fake = tf.sigmoid(c_val_batch_fake)
        d_val_synth = tf.sigmoid(c_val_synth)
        d_val_batch_synth = tf.sigmoid(c_val_batch_synth)

        val_bc_disc = tf.reduce_mean(d_val_fake)
        val_bc_gen = tf.reduce_mean(d_val_synth)
        val_bc_acc = tf.reduce_mean(tf.round(d_val_fake))

        val_bc_batch_disc = tf.reduce_mean(d_val_batch_fake)
        val_bc_batch_gen = tf.reduce_mean(d_val_batch_synth)
        val_bc_batch_acc = tf.reduce_mean(tf.round(d_val_batch_fake))

        val_synth_bc_auto = real_states_val[:,-1] + self.gen.state_recon(real_states_val[:,-1], z0) / np.sqrt(self.state_dim)
        val_bc_diff_auto = tf.sqrt(tf.reduce_sum((real_states_val[:,-1] - val_synth_bc_auto)**2, axis=-1))
        val_bc_diff = tf.sqrt(tf.reduce_sum((real_states_val[:,-1] - synth_bc_val)**2, axis=-1))
        val_bc_norm = tf.sqrt(tf.reduce_sum(real_states_val[:,-1]**2, axis=-1))
        val_bc_auto = tf.reduce_mean(val_bc_diff_auto / val_bc_norm)
        val_bc_stab = tf.reduce_mean(val_bc_diff / val_bc_norm)


        # Phase 4 training
        # Leak metrics
        leak_synth = self.leak(synth_states_val)
        val_leak_gen = tf.reduce_mean(leak_synth)
        val_leak_gen_rel = tf.reduce_mean(tf.maximum(leak_synth * (1 - leak_real), epsilon) /
                                          tf.maximum(leak_real + leak_synth - 2* leak_real * leak_synth, epsilon))

        

        # Generator metrics
        val_gen = tf.reduce_mean(self.disc(synth_states[:,-1]))
        val_gen_nll = -tf.reduce_mean(tf.log(self.disc(synth_states[:,-1] + 1e-12)))

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            print("Initializing variables")
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            global_epoch = 0
            nll = []
            start = time.time()

            # Phase 1 - scheduling
            if phase<=1:
                for skip_ind in range(skip_no):
                    epoch = 0
                    class_acc = 0.
                    mapper_acc = 0.
                    gen_comp_acc = 0.
                    mean_norm = 0.
                    temp_nll = float('inf')
                    while ((class_acc < .99 - skip_ind * .005) or
                           (gen_comp_acc < .94 - skip_ind * .005) or
                           (mapper_acc < .999)):
                        if (epoch + 1 > 50) and (class_acc > .9) and (skip_no + 1 > 5):
                            break
                        print('\nPhase 1, Generating %d words:' % (skip_ind+1))
                        print("Epoch:", epoch+1)
                        print('Global epoch:', global_epoch+1)
                        running_loss = 0.
                        running_acc = 0.
                        for _ in tqdm(range(batch_no_train), ascii=True,
                                      desc="Training"):
                            wat, why, _, _ = sess.run([loss_phase_1_class[skip_ind],
                                                          val_sched_train[skip_ind],
                                                          train_1_disc[skip_ind],
                                                          train_1_class[skip_ind]])
                            running_loss += wat
                            running_acc += why
                        running_loss /= batch_no_train
                        running_acc /= batch_no_train
                        class_acc = 0.
                        mapper_acc = 0.
                        temp_nll = 0.
                        gen_comp_acc = 0.
                        disc_acc = 0.
                        synth_score = 0.
                        real_score = 0.
                        rel_score = 0.
                        state_comp_acc = 0.
                        for _ in tqdm(range(batch_no_val), ascii=True,
                                      desc="Validating"):
                            temp = sess.run([val_sched_class[skip_ind],
                                             val_mapper,
                                             val_auto_nll,
                                             val_sched_comp[skip_ind],
                                             val_leak_sched_acc[skip_ind],
                                             val_leak_sched_synth[skip_ind],
                                             val_leak_real,
                                             val_leak_sched_rel[skip_ind],
                                             val_state_comp_sched[skip_ind]])
                            class_acc += temp[0]
                            mapper_acc += temp[1]
                            temp_nll += temp[2]
                            gen_comp_acc += temp[3]
                            disc_acc += temp[4]
                            synth_score += temp[5]
                            real_score += temp[6]
                            rel_score += temp[7]
                            state_comp_acc += temp[8]
                        class_acc /= batch_no_val
                        mapper_acc /= batch_no_val
                        temp_nll /= batch_no_val
                        gen_comp_acc /= batch_no_val
                        disc_acc /= batch_no_val
                        synth_score /= batch_no_val
                        real_score /= batch_no_val
                        rel_score /= batch_no_val
                        state_comp_acc /= batch_no_val
                        mean_norm = sess.run(val_norm)
                        print("Sched-synthetic word acc:", class_acc)
                        print("Sched-synthetic vector acc:", gen_comp_acc)
                        print("Sched-synthetic state acc:", state_comp_acc)
                        print("Mean embedding norm:", mean_norm)
                        print("Auto-synthetic word NLL:", temp_nll)
                        print("Mapper acc:", mapper_acc)
                        print("Leak disc acc:", disc_acc)
                        print("Auto-synthetic leak score:", synth_score)
                        print("Real leak score:", real_score)
                        print("Relativistic leak score:", rel_score)
                        print("Training sched loss:", running_loss)
                        print("Training acc:", running_acc)
                        print("Ellapsed time:", (time.time() - start) / 3600)
                        nll.append([global_epoch, temp_nll, 1])
                        global_epoch += 1
                        epoch += 1
                save_vars = {}
                for var in tf.trainable_variables():
                    save_vars[var.name] = sess.run(var).tolist()
                with open('saved/training/caption_train_1.json', 'w') as file:
                    json.dump(save_vars, file)

            # Phase 2 - autoencoding
            if phase <= 2:
                if phase == 2:
                    print("Loading saved model parameters")
                    with open('saved/training/caption_train_1.json', 'r') as file:
                        save_vars = json.load(file)
                    for var in tqdm(tf.trainable_variables()):
                        sess.run(tf.assign(var,
                                           np.array(save_vars[var.name])))
                epoch = 0
                class_acc = 0.
                mapper_acc = 0.
                gen_comp_acc = 0.
                mean_norm = 0.
                temp_nll = float('inf')
                while (class_acc < .92):
                    if (epoch + 1 > 100) and (class_acc >= .90):
                        break
                    print('\nPhase 2, Autoencoding')
                    print("Epoch:", epoch+1)
                    print('Global epoch:', global_epoch+1)
                    running_loss = 0.
                    running_acc = 0.
                    for _ in tqdm(range(batch_no_train), ascii=True,
                                  desc="Training"):
                        wat, why, _, _ = sess.run([loss_phase_2_class,
                                                   val_auto_train,
                                                   train_2_disc,
                                                   train_2_class])
                        running_loss += wat
                        running_acc += why
                    running_loss /= batch_no_train
                    running_acc /= batch_no_train
                    class_acc = 0.
                    mapper_acc = 0.
                    temp_nll = 0.
                    gen_comp_acc = 0.
                    disc_acc = 0.
                    synth_score = 0.
                    real_score = 0.
                    state_comp_acc = 0.
                    for _ in tqdm(range(batch_no_val), ascii=True,
                                  desc="Validating"):
                        temp = sess.run([val_auto_class,
                                         val_mapper,
                                         val_auto_nll,
                                         val_auto_comp,
                                         val_leak_auto_acc,
                                         val_leak_auto_synth,
                                         val_leak_real,
                                         val_leak_auto_rel,
                                         val_state_auto_comp])
                        class_acc += temp[0]
                        mapper_acc += temp[1]
                        temp_nll += temp[2]
                        gen_comp_acc += temp[3]
                        disc_acc += temp[4]
                        synth_score += temp[5]
                        real_score += temp[6]
                        rel_score += temp[7]
                        state_comp_acc += temp[8]
                    class_acc /= batch_no_val
                    mapper_acc /= batch_no_val
                    temp_nll /= batch_no_val
                    gen_comp_acc /= batch_no_val
                    disc_acc /= batch_no_val
                    synth_score /= batch_no_val
                    real_score /= batch_no_val
                    rel_score /= batch_no_val
                    state_comp_acc /= batch_no_val
                    mean_norm = sess.run(val_norm)
                    print("Auto-synthetic word acc:", class_acc)
                    print("Auto-synthetic vector acc:", gen_comp_acc)
                    print("Auto-synthetic state acc:", state_comp_acc)
                    print("Mean embedding norm:", mean_norm)
                    print("Auto-synthetic word NLL:", temp_nll)
                    print("Mapper acc:", mapper_acc)
                    print("Leak disc acc:", disc_acc)
                    print("Auto-synthetic leak score:", synth_score)
                    print("Real leak score:", real_score)
                    print("Relativistic leak score:", rel_score)
                    print("Training auto loss:", running_loss)
                    print("Training acc:", running_acc)
                    print("Ellapsed time:", (time.time() - start) / 3600)
                    nll.append([global_epoch, temp_nll, 1])
                    global_epoch += 1
                    epoch += 1
                save_vars = {}
                for var in tf.trainable_variables():
                    save_vars[var.name] = sess.run(var).tolist()
                with open('saved/training/caption_train_2.json', 'w') as file:
                    json.dump(save_vars, file)

            # Phase 3 - adversarial BC training
            if phase <= 3:
                if phase == 3:
                    print("Loading saved model parameters")
                    with open('saved/training/caption_train_2.json', 'r') as file:
                        save_vars = json.load(file)
                    for var in tqdm(tf.trainable_variables()):
                        try:
                            sess.run(tf.assign(var,
                                               np.array(save_vars[var.name])))
                        except:
                            pass
                epoch = 0
                auto_err = float('inf')
                while auto_err > .01:
                    print('\nPhase 3, Pre-training')
                    print('Epoch:', epoch+1)
                    print('Global epoch:', global_epoch+1)
                    for _ in tqdm(range(batch_no_train), ascii=True,
                                  desc='Training'):
                        sess.run(train_3_auto)
                        auto_err = 0.
                    for _ in tqdm(range(batch_no_val), ascii=True,
                                  desc='Validating'):
                        temp = sess.run(val_bc_auto)
                        auto_err += temp
                    auto_err /= batch_no_val
                    print("BC auto score:", 1 - auto_err)
                    global_epoch += 1
                    epoch += 1
                epoch = 0
                disc_acc = 0.
                synth_score = 0.
                real_score = 0.
                bc_auto_score = 0.
                mean_norm = 0.
                temp_nll = float('inf')
                real_score = 0.
                part_dist = 0.
                while ((real_score < .9) or
                       (np.abs(synth_score -.5) > .1) or
                       (epoch < 10)):
                    print('\nPhase 3, BC GAN')
                    print("Epoch:", epoch+1)
                    print('Global epoch:', global_epoch+1)
                    running_loss = 0.
                    running_acc = 0.
                    for _ in tqdm(range(batch_no_train), ascii=True,
                                  desc="Training Disc"):
                        sess.run([train_3_disc, train_3_auto])
                    for _ in tqdm(range(batch_no_train), ascii=True,
                                  desc="Training GEN"):
                        wat, _, _ = sess.run([loss_phase_3_gen,
                                              train_3_gen,
                                              train_3_auto])
                        running_loss += wat
                    running_loss /= batch_no_train
                    disc_acc = 0.
                    synth_score = 0.
                    real_score = 0.
                    batch_acc = 0.
                    synth_batch_score = 0.
                    real_batch_score = 0.
                    bc_auto_score = 0.
                    part_dist = 0.
                    bc_stab_score = 0.
                    for _ in tqdm(range(batch_no_val), ascii=True,
                                  desc="Validating"):
                        temp = sess.run([val_bc_acc,
                                         val_bc_gen,
                                         val_bc_disc,
                                         val_bc_batch_acc,
                                         val_bc_batch_gen,
                                         val_bc_batch_disc,
                                         val_bc_auto,
                                         val_particle_dist_ratio,
                                         val_bc_stab])
                        disc_acc += temp[0]
                        synth_score += temp[1]
                        real_score += temp[2]
                        batch_acc += temp[3]
                        synth_batch_score += temp[4]
                        real_batch_score += temp[5]
                        bc_auto_score += temp[6]
                        part_dist += temp[7]
                        bc_stab_score += temp[8]
                    disc_acc /= batch_no_val
                    synth_score /= batch_no_val
                    real_score /= batch_no_val
                    batch_acc /= batch_no_val
                    synth_batch_score /= batch_no_val
                    real_batch_score /= batch_no_val
                    bc_auto_score /= batch_no_val
                    bc_auto_score = 1 - bc_auto_score
                    part_dist /= batch_no_val
                    bc_stab_score /= batch_no_val
                    bc_stab_score = 1 - bc_stab_score
                    print("BC disc acc:", disc_acc)
                    print("BC real-fake score:", real_score)
                    print("BC synth-real score:", synth_score)
                    print("BC synth/real particle distance ratio:", part_dist)
                    print("BC auto score:", bc_auto_score)
                    print("BC stabilizer score:", bc_stab_score)
                    print("Training loss:", running_loss)
                    print("Ellapsed time:", (time.time() - start) / 3600)
                    nll.append([global_epoch, temp_nll, 1])
                    global_epoch += 1
                    epoch += 1
                save_vars = {}
                for var in tf.trainable_variables():
                    save_vars[var.name] = sess.run(var).tolist()
                with open('saved/training/caption_train_3.json', 'w') as file:
                    json.dump(save_vars, file)

            # Phase 4 - Generative
            if phase <= 4:
                if phase == 4:
                    print("Loading saved model parameters")
                    with open('saved/training/caption_train_3.json', 'r') as file:
                        save_vars = json.load(file)
                    for var in tqdm(tf.trainable_variables()):
                        try:
                            sess.run(tf.assign(var,
                                               np.array(save_vars[var.name])))
                        except:
                            pass
                epoch = 0
                class_acc = 0.
                mapper_acc = 0.
                gen_comp_acc = 0.
                mean_norm = 0.
                gen_rel_score = 0.
                temp_nll = float('inf')
                while ((class_acc < .90) or (gen_rel_score < .45)):
                    if ((epoch + 1 > 10) and (class_acc >= .89)):
                        break
                    print('\nPhase 4, Generative training')
                    print("Epoch:", epoch+1)
                    print('Global epoch:', global_epoch+1)
                    running_loss = 0.
                    running_acc = 0.
                    for _ in tqdm(range(batch_no_train), ascii=True,
                                  desc="Disc training"):
                        sess.run([train_4_disc, train_4_class, train_3_disc, train_3_auto])
                    for _ in tqdm(range(batch_no_train), ascii=True,
                                  desc="Training"):
                        wat, why, _, _, _, _ = sess.run([loss_phase_2_class,
                                                         val_auto_train,
                                                         train_4_gen,
                                                         train_4_class,
                                                         train_4_bc_gen,
                                                         train_3_auto])
                        running_loss += wat
                        running_acc += why
                    running_loss /= batch_no_train
                    running_acc /= batch_no_train
                    class_acc = 0.
                    mapper_acc = 0.
                    temp_nll = 0.
                    gen_comp_acc = 0.
                    disc_acc = 0.
                    synth_score = 0.
                    real_score = 0.
                    gen_score = 0.
                    gen_rel_score = 0.
                    rel_score = 0.
                    disc_bc_acc = 0.
                    real_bc_score = 0.
                    synth_bc_score = 0.
                    part_dist = 0.
                    for _ in tqdm(range(batch_no_val), ascii=True,
                                  desc="Validating"):
                        temp = sess.run([val_auto_class,
                                         val_mapper,
                                         val_auto_nll,
                                         val_auto_comp,
                                         val_leak_real,
                                         val_leak_gen,
                                         val_leak_gen_rel,
                                         val_leak_auto_rel,
                                         val_bc_acc,
                                         val_bc_gen,
                                         val_bc_disc,
                                         val_particle_dist_ratio])
                        class_acc += temp[0]
                        mapper_acc += temp[1]
                        temp_nll += temp[2]
                        gen_comp_acc += temp[3]
                        real_score += temp[4]
                        gen_score += temp[5]
                        gen_rel_score += temp[6]
                        rel_score += temp[7]
                        disc_bc_acc += temp[8]
                        real_bc_score += temp[10]
                        synth_bc_score += temp[9]
                        part_dist += temp[11]
                    class_acc /= batch_no_val
                    mapper_acc /= batch_no_val
                    temp_nll /= batch_no_val
                    gen_comp_acc /= batch_no_val
                    disc_acc /= batch_no_val
                    synth_score /= batch_no_val
                    real_score /= batch_no_val
                    gen_score /= batch_no_val
                    gen_rel_score /= batch_no_val
                    rel_score /= batch_no_val
                    disc_bc_acc /= batch_no_val
                    real_bc_score /= batch_no_val
                    synth_bc_score /= batch_no_val
                    part_dist /= batch_no_val
                    mean_norm = sess.run(val_norm)
                    print("Auto-synthetic word acc:", class_acc)
                    print("Auto-synthetic vector acc:", gen_comp_acc)
                    print("Mean embedding norm:", mean_norm)
                    print("Auto-synthetic word NLL:", temp_nll)
                    print("Mapper acc:", mapper_acc)
#                    print("Gen-synthetic leak score:", gen_score)
#                    print("Real leak score:", real_score)
                    print("Auto-relativistic leak score:", rel_score)
                    print("Relativistic leak score:", gen_rel_score)
                    print("BC disc acc:", disc_bc_acc)
                    print("BC real-fake score:", real_bc_score)
                    print("BC synth-real score:", synth_bc_score)
                    print("BC synth/real particle distance ratio:", part_dist)
                    print("Training auto loss:", running_loss)
                    print("Training acc:", running_acc)
                    print("Ellapsed time:", (time.time() - start) / 3600)
                    nll.append([global_epoch, temp_nll, 1])
                    global_epoch += 1
                    epoch += 1
                save_vars = {}
                for var in tf.trainable_variables():
                    save_vars[var.name] = sess.run(var).tolist()
                with open('saved/training/caption_train_4.json', 'w') as file:
                    json.dump(save_vars, file)

            bleu_scores = self.bleu_benchmark(val_set[0], sess)
            with open('tests/bleu_scores.csv','w') as file:
                writer = csv.writer(file)
                writer.writerow(bleu_scores)
            auto_real, auto_synth, auto_acc = self.autoencoder_benchmark(val_set[0], sess)
            with open('tests/auto_exs.csv','w') as file:
                writer = csv.writer(file)
                for i, row in enumerate(auto_real):
                    writer.writerow([row, auto_synth[i]])
            with open('tests/auto_score.csv','w') as file:
                writer = csv.writer(file)
                writer.writerow([auto_acc])
            synth_words = self.synth_benchmark(synth, sess)
            with open('tests/synth_words.csv','w') as file:
                writer = csv.writer(file)
                for row in synth_words:
                    writer.writerow([row])
            train_bleu = self.bleu_benchmark(train_set[0], sess)
            with open('tests/bleu_scores_train.csv','w') as file:
                writer = csv.writer(file)
                writer.writerow(train_bleu)
            
            
            coord.request_stop()
            coord.join(threads)
            save_vars = (tf.trainable_variables('embeddings') +
                         tf.trainable_variables('generator') +
                         tf.trainable_variables('word_classifier'))
            best_vars = [sess.run(var) for var in save_vars]
            sess.close()


        tf.reset_default_graph()
        with tf.variable_scope('generator'):
            self.gen = Gauge(self.state_dim, self.word_dim, self.latent_dim)
        with tf.variable_scope('word_classifier'):
            self.word_class = bn.Word_Class(self.word_dim, vocab_size)
        with tf.variable_scope('embeddings'):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                    shape=[vocab_size, self.word_dim],
                                    initializer=initializer)
        save_vars = (tf.trainable_variables('embeddings') +
                         tf.trainable_variables('generator') +
                         tf.trainable_variables('word_classifier'))
        with tf.Session() as sess:
            for i, var in enumerate(save_vars):
                sess.run(var.assign(best_vars[i]))
            saver = tf.train.Saver()
            saver.save(sess, 'saved/serving/caption_serve.ckpt')

    def autoencoder_benchmark(self, real_inds, sess):
        real_inds = tf.convert_to_tensor(real_inds)
        real = tf.nn.embedding_lookup(self.embedding_matrix, real_inds)
        real_words = vec2text(real, self.word_class, self.id_word, sess)
        real_words = [' '.join([word for word in sent if word != '#EoS']) for sent in real_words]
        synth, _, _ = self.gen.autoencode(real)
        synth_words = vec2text(synth, self.word_class, self.id_word, sess)
        synth_words = [' '.join([word for word in sent if word != '#EoS']) for sent in synth_words]
        synth_pdf = self.word_class(synth)
        acc = sess.run(word_class_acc(real_inds, synth_pdf))
        return real_words, synth_words, acc

    def synth_benchmark(self, synth, sess):
        synth_set = []
        for _ in range(10):
            synth_words = vec2text(synth, self.word_class, self.id_word, sess)
            synth_words = [' '.join([word for word in sent if word != '#EoS']) for sent in synth_words]
            synth_set += synth_words
        return synth_words

    def bleu_benchmark(self, real_inds, sess):
        real_inds = tf.convert_to_tensor(real_inds)
        real = tf.nn.embedding_lookup(self.embedding_matrix, real_inds)
        real_words = vec2text(real, self.word_class, self.id_word, sess)
        real_words = [[word for word in sent if word != '#EoS'] for sent in real_words]
        synth, _, _ = self.gen.autoencode(real)
        synth_words = vec2text(synth, self.word_class, self.id_word, sess)
        synth_words = [[word for word in sent if word != '#EoS'] for sent in synth_words]

        bleu2 = []
        bleu3 = []
        bleu4 = []
        bleu5 = []
        for i, ref in enumerate(real_words):
            # BLEU2
            bleu2.append(sentence_bleu([ref], synth_words[i], weights=(0,1,0,0,0),
                                       smoothing_function=SmoothingFunction().method1))
            # BLEU3
            bleu3.append(sentence_bleu([ref], synth_words[i], weights=(0,0,1,0,0),
                                       smoothing_function=SmoothingFunction().method1))
            # BLEU4
            bleu4.append(sentence_bleu([ref], synth_words[i], weights=(0,0,0,1,0),
                                       smoothing_function=SmoothingFunction().method1))
            # BLEU5
            bleu5.append(sentence_bleu([ref], synth_words[i], weights=(0,0,0,0,1),
                                       smoothing_function=SmoothingFunction().method1))
        bleu2_mean = np.mean(bleu2)
        bleu3_mean = np.mean(bleu3)
        bleu4_mean = np.mean(bleu4)
        bleu5_mean = np.mean(bleu5)
        return bleu2_mean, bleu3_mean, bleu4_mean, bleu5_mean
