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


def get_tf_sim_scorer(name, lang, sent_seq, doc_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(sent_seq, 5, 2).add_LSTM_input(doc_seq, 5, 2)
    graph.add_input(lang)
    graph.add_hidden(10, tf.nn.tanh).add_hidden(10, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.tanh)
    return graph.get_output()

def get_size_scorer(name, lang, sent_size, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_input(lang).add_input(sent_size)
    graph.add_LSTM_input(doc_size_seq, 5, 2)
    graph.add_hidden(10, tf.nn.tanh).add_hidden(10, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.tanh)
    return graph.get_output()

def get_position_scorer(name, lang, sent_pos, doc_size):
    graph = Scorer(name)
    graph.add_input(lang).add_input(sent_pos).add_input(doc_size)
    graph.add_hidden(10, tf.nn.tanh).add_hidden(10, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.tanh)
    return graph.get_output()

def get_language_scorer(name, doc_tf_seq, doc_sim_seq, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(doc_tf_seq, 5, 2)
    graph.add_LSTM_input(doc_sim_seq, 5, 2)
    graph.add_LSTM_input(doc_size_seq, 5, 2)
    graph.add_hidden(10, tf.nn.tanh).add_hidden(10, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.tanh)
    return graph.get_output()

def get_sentence_scorer(name, lang, tfreq, sim, size, pos):
    graph = Scorer(name)
    graph.add_input(lang)
    graph.add_input(tfreq)
    graph.add_input(sim)
    graph.add_input(size)
    graph.add_input(pos)
    graph.add_hidden(10, tf.nn.tanh).add_hidden(10, tf.nn.tanh) # 2 hidden layers
    graph.add_output(1, tf.nn.tanh)
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
        self.sent_sim_seq = tf.plalder(tf.float32, shape=[None,None,1], name="sent_tf_seq_in")
        # similarities of this sentence with others
        self.sent_sim_seq = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_sim_seq_in")
        # sentence size
        self.sent_size = tf.placeholder(tf.float32, shape=[None,1], name="sent_size_in")
        # sentence position
        self.sent_pos = tf.placeholder(tf.float32, shape=[None,1], name="sent_pos_in")

        self.rouge_1 = tf.placeholder(tf.float32, shape=[None,1], name="rouge_1_out")


        #          Model
        # =====================
        self.lang_scorer = get_language_scorer("lang_scorer", doc_tf_seq, doc_sim_seq, doc_size_seq)

        self.tf_scorer = get_tf_sim_scorer("tf_scorer", lang_scorer, sent_tf_seq, doc_tf_seq)
        self.sim_scorer = get_tf_sim_scorer("sim_scorer", lang_scorer, sent_sim_seq, doc_sim_seq)
        self.size_scorer = get_size_scorer("size_scorer", lang_scorer, sent_size, doc_size_seq)
        self.pos_scorer = get_position_scorer("pos_scorer", lang_scorer, sent_pos, doc_size)

        self.graph = get_sentence_scorer("sent_scorer", lang_scorer, tf_scorer, sim_scorer, size_scorer, pos_scorer)



        self.cost = tf.losses.mean_squared_error(self.rouge_1, self.graph)

        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cost)


    def train(batch):
        
