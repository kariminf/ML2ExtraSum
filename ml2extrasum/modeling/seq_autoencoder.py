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

from ml2extrasum.scoring.scorer import Scorer
from ml2extrasum.scoring.seq_scorer import SeqScorer
from ml2extrasum.scoring.seq_scorer import transform_output

import tensorflow as tf

LSTM_NUM_LAYERS = 2
LSTM_HID_SIZE = 30
ENC_NUM_OUT = 5

def get_encoder(name, seq):
    graph = SeqScorer(name)
    #input, hidden_size, num_layers = 1, num_proj, activation=tf.nn.tanh
    graph.add_LSTM_input(seq, LSTM_HID_SIZE, LSTM_NUM_LAYERS)
    graph.add_layer(ENC_NUM_OUT, tf.nn.tanh)
    return graph.get_output()

def get_decoder(name, vec, seq_len):
    graph = Scorer(name)
    graph.add_input(vec)
    graph.add_layer(ENC_NUM_OUT, tf.nn.tanh)
    vec2 = graph.get_output()
    #initial_state_dec = tuple([(vec2, vec2)] * LSTM_NUM_LAYERS)

    batch_size = tf.shape(vec)[0]

    dec_inputs = tf.zeros([batch_size, seq_len, 1])
    """
    dec_inputs = [
    [[1.0], [2.1], [3.2], [4.3]],
    [[5.4], [6.5], [7.6], [8.7]]
    ]
    dec_inputs = tf.convert_to_tensor(dec_inputs, dtype=tf.float32)
    """
    #dec_inputs = tf.mul([tf.zeros([batch_size, 1])], seq_len)
    #print dec_inputs

    cell_dec = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LSTMCell(LSTM_HID_SIZE) for _ in range(LSTM_NUM_LAYERS)])

    initial_state_dec = cell_dec.zero_state(batch_size, dtype=tf.float32)

    output, _ = tf.nn.dynamic_rnn(cell_dec, inputs=dec_inputs, initial_state=initial_state_dec, dtype=tf.float32)

    #batch_range = tf.range(batch_size)
    #seq_range = tf.range(seq_len)
    #indices = tf.stack([batch_range, seq_range, [LSTM_HID_SIZE - 1]], axis=0)

    return output[:,:, -1]


class SeqAutoEncoder(Model):

    def __init__(self, name, seq):
        super(SeqAutoEncoder, self).__init__()

        #sequence length
        self.seq_len = tf.shape(seq)[1]
        #self.seq_len = 4

        self.latent = get_encoder(name + ".encoder", seq)
        self.graph = get_decoder(name + ".decoder", self.latent, self.seq_len)

    def get_latent(self):
        return self.latent
