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


TRAIN_ITER = 2
LEARNING_RATE = 0.05

def repeat_vector(vector, nbr):
    return [vector] * nbr

def get_tf_sim_scorer(name, lang, sent_seq, doc_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(sent_seq, 50, 3).add_LSTM_input(doc_seq, 50, 3)
    graph.add_input(lang)
    graph.add_hidden(100, tf.nn.relu).add_hidden(100, tf.nn.relu) # 2 hidden layers
    graph.add_output(1, tf.nn.softmax)
    return graph.get_output()

def get_size_scorer(name, lang, sent_size, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_input(lang).add_input(sent_size)
    graph.add_LSTM_input(doc_size_seq, 50, 3)
    graph.add_hidden(100, tf.nn.relu).add_hidden(100, tf.nn.relu) # 2 hidden layers
    graph.add_output(1, tf.nn.softmax)
    return graph.get_output()

def get_position_scorer(name, lang, sent_pos, doc_size):
    graph = Scorer(name)
    graph.add_input(lang).add_input(sent_pos).add_input(doc_size)
    graph.add_hidden(100, tf.nn.tanh).add_hidden(100, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.softmax)
    return graph.get_output()

def get_language_scorer(name, doc_tf_seq, doc_sim_seq, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(doc_tf_seq, 50, 3)
    graph.add_LSTM_input(doc_sim_seq, 50, 3)
    graph.add_LSTM_input(doc_size_seq, 50, 3)
    graph.add_hidden(100, tf.nn.tanh).add_hidden(100, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.softmax)
    return graph.get_output()

def get_sentence_scorer(name, lang, tfreq, sim, size, pos):
    graph = Scorer(name)
    graph.add_input(lang)
    graph.add_input(tfreq)
    graph.add_input(sim)
    graph.add_input(size)
    graph.add_input(pos)
    graph.add_hidden(100, tf.nn.tanh).add_hidden(100, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.softmax)
    return graph.get_output()


class StatNet(Model):

    def __init__(self):
        super(StatNet, self).__init__()

        #       Inputs holders
        # =========================
        # term frequencies in document
        self.doc_tf_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_tf_seq_in")
        # all sentences similarities in a document
        self.doc_sim_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_sim_seq_in")
        # all sentences sizes in a document
        self.doc_size_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_size_seq_in")
        # document size
        self.doc_size = tf.placeholder(tf.float32, shape=[None,1], name="doc_size_in")
        # term frequencies (in the document) of a sentence
        self.sent_tf_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_tf_seq_in")
        # similarities of this sentence with others
        self.sent_sim_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_tf_seq_in")
        # similarities of this sentence with others
        self.sent_sim_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_sim_seq_in")
        # sentence size
        self.sent_size = tf.placeholder(tf.float32, shape=[None,1], name="sent_size_in")
        # sStatNetentence position
        self.sent_pos = tf.placeholder(tf.float32, shape=[None,1], name="sent_pos_in")

        self.rouge_1 = tf.placeholder(tf.float32, shape=[None,1], name="rouge_1_out")


        #          Model
        # =====================
        self.lang_scorer = get_language_scorer("lang_scorer", self.doc_tf_seq, self.doc_sim_seq, self.doc_size_seq)

        self.tf_scorer = get_tf_sim_scorer("tf_scorer", self.lang_scorer, self.sent_tf_seq, self.doc_tf_seq)
        self.sim_scorer = get_tf_sim_scorer("sim_scorer", self.lang_scorer, self.sent_sim_seq, self.doc_sim_seq)
        self.size_scorer = get_size_scorer("size_scorer", self.lang_scorer, self.sent_size, self.doc_size_seq)
        self.pos_scorer = get_position_scorer("pos_scorer", self.lang_scorer, self.sent_pos, self.doc_size)

        self.graph = get_sentence_scorer("sent_scorer", self.lang_scorer, self.tf_scorer, self.sim_scorer, self.size_scorer, self.pos_scorer)

        self.cost = tf.losses.mean_squared_error(self.rouge_1, self.graph)
        #self.cost = tf.losses.sigmoid_cross_entropy(self.rouge_1, self.graph)
        #self.cost = tf.reduce_mean(tf.losses.mean_squared_error(self.rouge_1, self.graph))


        #self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cost)
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

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
        scores["lang"] = lang.tolist()
        scores["tf"] = tfreq.tolist()
        scores["sim"] = sim.tolist()
        scores["pos"] = pos.tolist()
        scores["sent"] = sent.tolist()

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
