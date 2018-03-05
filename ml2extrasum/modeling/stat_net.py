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

def get_tf_sim_scorer(name, lang, sent_seq, doc_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(sent_seq, 5, 2).add_LSTM_input(doc_seq, 5, 2)
    graph.add_input(lang)
    graph.add_layer(10, tf.nn.tanh).add_layer(10, tf.nn.tanh) # 2 hidden layers
    graph.add_layer(1, tf.nn.tanh)
    return graph.get_output()

def get_size_scorer(name, lang, sent_size, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_input(lang).add_input(sent_size)
    graph.add_LSTM_input(doc_size_seq, 5, 2)
    graph.add_layer(10, tf.nn.tanh).add_layer(10, tf.nn.tanh) # 2 hidden layers
    graph.add_layer(1, tf.nn.tanh)
    return graph.get_output()

def get_position_scorer(name, lang, sent_pos, doc_size):
    graph = Scorer(name)
    graph.add_input(lang).add_input(sent_pos).add_input(doc_size)
    graph.add_layer(10, tf.nn.tanh).add_layer(10, tf.nn.tanh) # 2 hidden layers
    graph.add_layer(1, tf.nn.tanh)
    return graph.get_output()

def get_language_scorer(name, doc_tf_seq, doc_sim_seq, doc_size_seq):
    graph = SeqScorer(name)
    graph.add_LSTM_input(doc_tf_seq, 5, 2)
    graph.add_LSTM_input(doc_sim_seq, 5, 2)
    graph.add_LSTM_input(doc_size_seq, 5, 2)
    graph.add_layer(10, tf.nn.tanh).add_layer(10, tf.nn.tanh) # 2 hidden layers
    graph.add_layer(1, tf.nn.tanh)
    return graph.get_output()

def get_sentence_scorer(name, lang, tfreq, sim, size, pos):
    graph = Scorer(name)
    graph.add_input(lang)
    graph.add_input(tfreq)
    graph.add_input(sim)
    graph.add_input(size)
    graph.add_input(pos)
    graph.add_layer(10, tf.nn.tanh).add_layer(10, tf.nn.tanh) # 2 hidden layers
    graph.add_layer(1, tf.nn.tanh)
    return graph.get_output()


class StatNet(Model):

    def __init__(self,
                doc_tf_seq,
                doc_sim_seq,
                doc_size_seq,
                doc_size,
                sent_tf_seq,
                sent_sim_seq,
                sent_size,
                sent_pos):
        super(StatNet, self).__init__()

        lang_scorer = get_language_scorer("lang_scorer", doc_tf_seq, doc_sim_seq, doc_size_seq)

        tf_scorer = get_tf_sim_scorer("tf_scorer", lang_scorer, sent_tf_seq, doc_tf_seq)
        sim_scorer = get_tf_sim_scorer("sim_scorer", lang_scorer, sent_sim_seq, doc_sim_seq)
        size_scorer = get_size_scorer("size_scorer", lang_scorer, sent_size, doc_size_seq)
        pos_scorer = get_position_scorer("pos_scorer", lang_scorer, sent_pos, doc_size)

        self.graph = get_sentence_scorer("sent_scorer", lang_scorer, tf_scorer, sim_scorer, size_scorer, pos_scorer)
