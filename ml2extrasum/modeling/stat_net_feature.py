#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2018 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
#  2018	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
sys.path.insert(0,'../')

from model import Model

import tensorflow as tf

from scoring.scorer import Scorer
from scoring.seq_scorer import SeqScorer
from preprocessing.filter import Filter

HIDDEN_ACT = tf.nn.relu


def repeat_vector(vector, nbr):
    return [vector] * nbr

def get_tf_sim_scorer(name, lang, sent_seq, doc_seq):
    with tf.name_scope(name):
        with tf.name_scope("features"):

            with tf.name_scope("prepare"):
                sent_seq_len = tf.to_float(tf.count_nonzero(sent_seq, axis=1)) + 1.0
                doc_seq_len = tf.to_float(tf.count_nonzero(doc_seq, axis=1)) + 1.0

            with tf.name_scope("S2Dsum"):
                # sentence to document sum normalization (S2Dsum)
                # S2Dsum(i) = (|fD| * sum (fi))/(|fi| * sum (fD))
                # fk can be words tf in k, similarities in k, etc.
                sent_sum = tf.multiply(doc_seq_len, tf.reduce_sum(sent_seq, axis=1))
                doc_sum = tf.multiply(sent_seq_len, tf.reduce_sum(doc_seq, axis=1))
                s2dsum = tf.div(sent_sum, doc_sum)
                #s2dsum = tf.stop_gradient(s2dsum)

            with tf.name_scope("S2Dmxmn"):
                # sentence to document intervall normalization (S2Dmxmn)
                # S2Dmean(i) = mean(f(i))/mean(f(D))
                # f(k) can be words tf, sentences sim, etc.
                sent_max = tf.reduce_max(sent_seq, axis=1)
                sent_min = tf.reduce_min(sent_seq, axis=1)
                doc_max = tf.reduce_max(doc_seq, axis=1)
                doc_min = tf.reduce_min(doc_seq, axis=1)
                sent_diff = tf.multiply(doc_seq_len, (sent_max - sent_min + 1.0))
                doc_diff = tf.multiply(sent_seq_len, (doc_max - doc_min + 1.0))
                s2dmxmn = tf.div(sent_diff, doc_diff)
                #s2dmxmn = tf.stop_gradient(s2dmxmn)


        estim = SeqScorer(name + "_estim")
        estim.add_LSTM_input(sent_seq, 10, 1, 2).add_LSTM_input(doc_seq, 10, 1, 2)
        estim.add_hidden(5, HIDDEN_ACT)
        estim.add_output(1, tf.nn.sigmoid)
        estim = estim.get_output()

        graph = Scorer("scorer")
        graph.add_input(lang).add_input(estim).add_input(s2dsum).add_input(s2dmxmn)
        graph.add_hidden(20, HIDDEN_ACT)#.add_hidden(10, HIDDEN_ACT) # 2 hidden layers
        graph.add_output(1, tf.nn.sigmoid)
        return graph.get_output()

def get_size_scorer(name, lang, sent_size, doc_size_seq):
    with tf.name_scope(name):
        with tf.name_scope("features"):

            with tf.name_scope("prepare"):
                doc_maxsize = tf.reduce_max(doc_size_seq, axis=1, name="DmaxSIZE")
                doc_meansize = tf.reduce_mean(doc_size_seq, axis=1, name="DmeanSIZE")

                max_max = doc_maxsize * 0.7
                max_mean = doc_meansize * 1.3

                min_max = doc_maxsize * 0.3
                min_mean = doc_meansize * 0.7

                ones = tf.ones(tf.shape(sent_size))
                zeros = tf.zeros(tf.shape(sent_size))


            # Maximum normalization (MxN)
            # MxN = (Lmax - Li)/Lmax if Li <= Lmax; 1 otherwise
            # ==================================================
            with tf.name_scope("MxNMax"):
            # Maximum normalization based on maximum length
                mxn = (max_max - sent_size)/max_max
                mxnmax = tf.where(sent_size <= max_max, ones, mxn)
                #mxnmax = tf.stop_gradient(mxnmax)
            with tf.name_scope("MxNMean"):
            # Maximum normalization based on mean length
                mxn = (max_mean - sent_size)/max_mean
                mxnmean = tf.where(sent_size <= max_mean, ones, mxn)
                #mxnmean = tf.stop_gradient(mxnmean)

            # Minimum normalization (MnN)
            # MnN = (Li - Lmin)/Li if Li >= Lmin; 0 otherwise
            # (Li - Lmin)/(Li + 0.1 ) in case Li = 0
            # ==================================================
            with tf.name_scope("MnNMax"):
            # Minimum normalization based on maximum length
                mnn = (sent_size - min_max)/(sent_size + 0.1)
                mnnmax = tf.where(sent_size >= min_max, zeros, mnn)
                #mnnmax = tf.stop_gradient(mnnmax)
            with tf.name_scope("MnNMean"):
            # Minimum normalization based on mean length
                mnn = (sent_size - min_mean)/(sent_size + 0.1)
                mnnmean = tf.where(sent_size >= min_mean, zeros, mnn)
                #mnnmean = tf.stop_gradient(mnnmean)

        estim = SeqScorer(name + "_estim")
        estim.add_input(sent_size).add_LSTM_input(doc_size_seq, 10, 1, 2)
        estim.add_hidden(5, HIDDEN_ACT)
        estim.add_output(1, tf.nn.sigmoid)
        estim = estim.get_output()

        graph = SeqScorer("scorer")
        graph.add_input(lang).add_input(estim)
        graph.add_input(mxnmax).add_input(mxnmean)
        graph.add_input(mnnmax).add_input(mnnmean)
        graph.add_hidden(20, HIDDEN_ACT)#.add_hidden(50, HIDDEN_ACT) # 2 hidden layers
        graph.add_output(1, tf.nn.sigmoid)
        return graph.get_output()

def get_position_scorer(name, lang, sent_pos, doc_size):
    with tf.name_scope(name):
        with tf.name_scope("features"):
            with tf.name_scope("DP"):
                # We add 1 to sent_pos because positions start from 0
                # Direct proportion (DP)
                # dp(i) = (n - i + 1)/n | i: sent pos, n: doc size
                dp = tf.div((doc_size - sent_pos), doc_size)
                #dp = tf.stop_gradient(dp)
            with tf.name_scope("IP"):
                # Inverse proportion (IP)
                # ip(i) = 1/i
                ip = tf.div(1.0, (sent_pos + 1.0))
                #ip = tf.stop_gradient(ip)
            with tf.name_scope("PP"):
                # Position proportion (PP)
                # pp(i) = 1/(n-i+1)
                pp = tf.div(1.0, (doc_size - sent_pos))
                #pp = tf.stop_gradient(pp)
            with tf.name_scope("GS"):
                # Geometric sequence (GS)
                # gs(i) = (1/2)^{i-1}
                gs = tf.pow(0.5, sent_pos)
                #gs = tf.stop_gradient(gs)
            with tf.name_scope("MP"):
                # Max proportion (MP)
                # mp(i) = max(1/i, 1/(n-i+1))
                mp = tf.maximum(ip, pp)
                #mp = tf.stop_gradient(mp)

        estim = Scorer(name + "_estim")
        estim.add_input(sent_pos).add_input(doc_size)
        estim.add_hidden(5, HIDDEN_ACT)
        estim.add_output(1, tf.nn.sigmoid)
        estim = estim.get_output()

        graph = Scorer("scorer")
        graph.add_input(lang).add_input(estim)
        graph.add_input(dp).add_input(ip).add_input(pp)
        graph.add_input(gs).add_input(mp)
        graph.add_hidden(20, HIDDEN_ACT)#.add_hidden(10, HIDDEN_ACT) # 2 hidden layers
        graph.add_output(1, tf.nn.sigmoid)
        return graph.get_output()

def get_language_scorer(name, doc_tf_seq, doc_sim_seq, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(doc_tf_seq, 10, 2)
    graph.add_LSTM_input(doc_sim_seq, 10, 2)
    graph.add_LSTM_input(doc_size_seq, 10, 2)
    graph.add_hidden(20, HIDDEN_ACT)
    graph.add_output(2, tf.nn.sigmoid)
    return graph.get_output()

def get_sentence_scorer(name, lang, tfreq, sim, size, pos):
    graph = Scorer(name)
    graph.add_input(lang)
    graph.add_input(tfreq)
    graph.add_input(sim)
    graph.add_input(size)
    graph.add_input(pos)
    graph.add_hidden(20, HIDDEN_ACT)#.add_hidden(5, HIDDEN_ACT) # 2 hidden layers
    graph.add_output(1, tf.nn.sigmoid)
    return graph.get_output()


class StatNet(Model):

    def __init__(self, learn_rate=0.05, cost_fct=tf.losses.mean_squared_error, opt_fct=tf.train.GradientDescentOptimizer):
        super(StatNet, self).__init__(learn_rate, cost_fct, opt_fct)

        #self.lang_scores = {}
        #self.current_lang = ""

        #       Inputs
        # ==============
        # term frequencies in document
        self.doc_tf_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_tf_seq")
        # all sentences similarities in a document
        self.doc_sim_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_sim_seq")
        # all sentences sizes in a document
        self.doc_size_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_size_seq")
        # document size
        self.doc_size = tf.placeholder(tf.float32, shape=[None,1], name="doc_size")
        # term frequencies (in the document) of a sentence
        self.sent_tf_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_tf_seq")
        # similarities of this sentence with others
        self.sent_sim_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_sim_seq")
        # sentence size
        self.sent_size = tf.placeholder(tf.float32, shape=[None,1], name="sent_size")
        # sStatNetentence position
        self.sent_pos = tf.placeholder(tf.float32, shape=[None,1], name="sent_pos")

        self.rouge_1 = tf.placeholder(tf.float32, shape=[None,1], name="rouge_1")

        # Preprocessing
        # ============================
        with tf.name_scope("preprocess") as scope:

            # term frequencies in document
            doc_tf_seq_norm = Filter(self.doc_tf_seq, "doc_tf_seq").get_graph()

            # all sentences similarities in a document
            doc_sim_seq_norm = Filter(self.doc_sim_seq, "doc_sim_seq").get_graph()

            # all sentences sizes in a document
            doc_size_seq_norm = Filter(self.doc_size_seq, "doc_size_seq").get_graph()

            # document size
            # Can't be normalized (for now) because a document is a batch
            doc_size_norm = self.doc_size

            # term frequencies (in the document) of a sentence
            sent_tf_seq_norm = Filter(self.sent_tf_seq, "sent_tf_seq").get_graph()

            # similarities of this sentence with others
            sent_sim_seq_norm = Filter(self.sent_sim_seq, "sent_sim_seq").get_graph()

            # sentence size
            sent_size_norm = self.sent_size

            # sentence position
            # No normalization because of doc_size
            sent_pos_norm = self.sent_pos


        #          Scorers
        # =====================
        self.lang_scorer = get_language_scorer("lang", doc_tf_seq_norm, doc_sim_seq_norm, doc_size_seq_norm)

        self.tf_scorer = get_tf_sim_scorer("tf", self.lang_scorer, sent_tf_seq_norm, doc_tf_seq_norm)
        self.sim_scorer = get_tf_sim_scorer("sim", self.lang_scorer, sent_sim_seq_norm, doc_sim_seq_norm)
        self.size_scorer = get_size_scorer("size", self.lang_scorer, sent_size_norm, doc_size_seq_norm)
        self.pos_scorer = get_position_scorer("pos", self.lang_scorer, sent_pos_norm, doc_size_norm)

        self.graph = get_sentence_scorer("sent", self.lang_scorer, self.tf_scorer, self.sim_scorer, self.size_scorer, self.pos_scorer)

        #          Training
        # =====================

        """
        with tf.name_scope("cost_function") as self.scope:
            self.lang_score = tf.Variable([0, 0], dtype=tf.float32, name="lang_tmp_score")
            # cost function
            self.cost1 = self.cost_fct(self.rouge_1, self.graph)

            lang_score = self.lang_scorer[0,:]

            self.cost2 = self.cost_fct(lang_score, self.lang_score)

            self.cost = self.cost1 + self.cost2

            #Save state of the language
            self.lang_scores[self.current_lang] = lang_score
            #Save state of the document of the same language
            self.lang_score = tf.assign(self.lang_score, lang_score)
        """

        self.cost = self.cost_fct(self.rouge_1, self.graph)

        # cost optimization
        self.train_step = self.opt_fct(self.learn_rate).minimize(self.cost)

        #      Initializing
        # =====================

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    """
    def init_lang_score (self, lang):
        self.current_lang = lang
        if lang in self.lang_scores:
            lang_score = self.lang_scores[lang]
        else:
            lang_score = [0, 0]

        self.lang_score = tf.assign(self.lang_score, lang_score)
    """

    def train(self, doc_data):
        nbr_sents = doc_data["nbr_sents"]
        feed = {
        self.doc_tf_seq : repeat_vector(doc_data["doc_tf_seq"], nbr_sents),
        self.doc_sim_seq : repeat_vector(doc_data["doc_sim_seq"], nbr_sents),
        self.doc_size_seq : repeat_vector(doc_data["doc_size_seq"], nbr_sents),
        self.doc_size : repeat_vector([nbr_sents], nbr_sents),
        self.sent_tf_seq : doc_data["sent_tf_seq"],
        self.sent_sim_seq : doc_data["sent_sim_seq"],
        self.sent_size : doc_data["sent_size"],
        self.sent_pos : doc_data["sent_pos"],
        self.rouge_1 : doc_data["rouge_1"]
        }
        _, cst = self.sess.run([self.train_step, self.cost], feed_dict=feed)
        return cst

    def test(self, doc_data):
        nbr_sents = doc_data["nbr_sents"]
        feed = {
        self.doc_tf_seq : repeat_vector(doc_data["doc_tf_seq"], nbr_sents),
        self.doc_sim_seq : repeat_vector(doc_data["doc_sim_seq"], nbr_sents),
        self.doc_size_seq : repeat_vector(doc_data["doc_size_seq"], nbr_sents),
        self.doc_size : repeat_vector([nbr_sents], nbr_sents),
        self.sent_tf_seq : doc_data["sent_tf_seq"],
        self.sent_sim_seq : doc_data["sent_sim_seq"],
        self.sent_size : doc_data["sent_size"],
        self.sent_pos : doc_data["sent_pos"],
        self.rouge_1 : doc_data["rouge_1"]
        }

        lang, tfreq, sim, size, pos, sent, cst = self.sess.run([self.lang_scorer, self.tf_scorer, self.sim_scorer, self.size_scorer, self.pos_scorer, self.graph, self.cost], feed_dict=feed)

        scores = {}
        scores["cost"] = cst
        scores["lang"] = lang[0,:].tolist()
        scores["tf"] = tfreq.flatten().tolist()
        scores["sim"] = sim.flatten().tolist()
        scores["pos"] = pos.flatten().tolist()
        scores["size"] = size.flatten().tolist()
        scores["sent"] = sent.flatten().tolist()

        return scores

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Model saved in file: %s" % save_path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def get_session(self):
        return self.sess
