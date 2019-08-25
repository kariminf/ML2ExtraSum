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
from transforming.filter import Filter

HIDDEN_ACT = tf.nn.relu


def repeat_vector(vector, nbr):
    return [vector] * nbr

def get_sentence_scorer(name, lang, tfreq, sim, size, pos):
    graph = Scorer(name)
    graph.add_input(lang)
    graph.add_input(tfreq)
    graph.add_input(sim)
    graph.add_input(size)
    graph.add_input(pos)
    graph.add_hidden(50, HIDDEN_ACT).add_hidden(20, HIDDEN_ACT) # 2 hidden layers
    graph.add_output(1, tf.nn.sigmoid)
    return graph.get_output()

def get_tf_sim_preprocess(name, sent_seq, doc_seq):
    with tf.name_scope(name):

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
            # sentence to document interval normalization (S2Dmxmn)
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
    return s2dsum, s2dmxmn

def create_scorer(name, inputs, nbr_outs):
    graph = Scorer(name)
    for input in inputs:
        graph.add_input(input)
    graph.add_hidden(50, HIDDEN_ACT)#.add_hidden(10, HIDDEN_ACT) # 2 hidden layers
    graph.add_output(nbr_outs, tf.nn.sigmoid)
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

            with tf.name_scope("sentPos"):
                with tf.name_scope("DP"):
                    # We add 1 to sent_pos because positions start from 0
                    # Direct proportion (DP)
                    # dp(i) = (n - i + 1)/n | i: sent pos, n: doc size
                    dp = tf.div((self.doc_size - self.sent_pos), self.doc_size)
                    #dp = tf.stop_gradient(dp)
                with tf.name_scope("IP"):
                    # Inverse proportion (IP)
                    # ip(i) = 1/i
                    ip = tf.div(1.0, (self.sent_pos + 1.0))
                    #ip = tf.stop_gradient(ip)
                with tf.name_scope("GS"):
                    # Geometric sequence (GS)
                    # gs(i) = (1/2)^{i-1}
                    gs = tf.pow(0.5, self.sent_pos)
                    #gs = tf.stop_gradient(gs)

            s2dsum_tf, s2dmxmn_tf = get_tf_sim_preprocess("tf", self.sent_tf_seq, self.doc_tf_seq)

            s2dsum_sim, s2dmxmn_sim = get_tf_sim_preprocess("sim", self.sent_sim_seq, self.doc_sim_seq)

            with tf.name_scope("sentSize"):

                with tf.name_scope("prepare"):
                    doc_maxsize = tf.reduce_max(self.doc_size_seq, axis=1, name="DmaxSIZE")
                    doc_meansize = tf.reduce_mean(self.doc_size_seq, axis=1, name="DmeanSIZE")

                    max_max = doc_maxsize * 0.7
                    max_mean = doc_meansize * 1.3

                    min_max = doc_maxsize * 0.3
                    min_mean = doc_meansize * 0.7

                    ones = tf.ones(tf.shape(self.sent_size))
                    zeros = tf.zeros(tf.shape(self.sent_size))


                # Maximum normalization (MxN)
                # MxN = (Lmax - Li)/Lmax if Li <= Lmax; 1 otherwise
                # ==================================================
                with tf.name_scope("MxNMax"):
                # Maximum normalization based on maximum length
                    mxn = (max_max - self.sent_size)/max_max
                    mxnmax = tf.where(self.sent_size > max_max, ones, mxn)
                    #mxnmax = tf.stop_gradient(mxnmax)
                with tf.name_scope("MxNMean"):
                # Maximum normalization based on mean length
                    mxn = (max_mean - self.sent_size)/max_mean
                    mxnmean = tf.where(self.sent_size > max_mean, ones, mxn)
                    #mxnmean = tf.stop_gradient(mxnmean)

                # Minimum normalization (MnN)
                # MnN = (Li - Lmin)/Li if Li >= Lmin; 0 otherwise
                # (Li - Lmin)/(Li + 0.1 ) in case Li = 0
                # ==================================================
                with tf.name_scope("MnNMax"):
                # Minimum normalization based on maximum length
                    mnn = (self.sent_size - min_max)/(self.sent_size + 0.1)
                    mnnmax = tf.where(self.sent_size < min_max, zeros, mnn)
                    #mnnmax = tf.stop_gradient(mnnmax)
                with tf.name_scope("MnNMean"):
                # Minimum normalization based on mean length
                    mnn = (self.sent_size - min_mean)/(self.sent_size + 0.1)
                    mnnmean = tf.where(self.sent_size < min_mean, zeros, mnn)
                    #mnnmean = tf.stop_gradient(mnnmean)
            with tf.name_scope("docTF"):
                max_doc_tf = tf.reduce_max(self.doc_tf_seq, axis=1)
                mean_doc_tf = tf.reduce_mean(self.doc_tf_seq, axis=1)
                doc_tf_perc = mean_doc_tf / (max_doc_tf + 0.1)

            with tf.name_scope("docSim"):
                max_doc_sim = tf.reduce_max(self.doc_sim_seq, axis=1)
                mean_doc_sim = tf.reduce_mean(self.doc_sim_seq, axis=1)
                doc_sim_perc = mean_doc_sim / (max_doc_sim + 0.1)

            with tf.name_scope("docSentSize"):
                max_doc_size = tf.reduce_max(self.doc_size_seq, axis=1)
                mean_doc_size = tf.reduce_mean(self.doc_size_seq, axis=1)
                doc_size_perc = mean_doc_size / (max_doc_size + 0.1)

        #doc_tf_seq, doc_sim_seq, doc_size_seq
        #          Scorers
        # =====================
        self.lang_scorer = create_scorer("lang", [
                doc_tf_perc, max_doc_sim, mean_doc_sim,
                doc_sim_perc, doc_size_perc
                ],2)

        self.tf_scorer = create_scorer("tf", [self.lang_scorer,s2dsum_tf, s2dmxmn_tf],1)
        self.sim_scorer = create_scorer("sim", [self.lang_scorer,s2dsum_sim, s2dmxmn_sim],1)
        self.size_scorer = create_scorer("size", [self.lang_scorer,mxnmax, mxnmean, mnnmax, mnnmean],1)
        self.pos_scorer = create_scorer("pos", [self.lang_scorer,dp, ip, gs],1)

        self.graph = get_sentence_scorer("sent", self.lang_scorer, self.tf_scorer, self.sim_scorer, self.size_scorer, self.pos_scorer)

        self.cost = self.cost_fct(self.rouge_1, self.graph)

        # cost optimization
        self.train_step = self.opt_fct(self.learn_rate).minimize(self.cost)

        #      Initializing
        # =====================
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

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
