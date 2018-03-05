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

import tensorflow as tf
#from deepmultilsum.scoring.scorer import Scorer
from scorer import Scorer

def transform_output(net):
    batch_size = tf.shape(net)[0]
    max_length = tf.shape(net)[1]
    out_size = int(net.get_shape()[2])
    index = tf.range(0, batch_size) * max_length
    flat = tf.reshape(net, [-1, out_size])
    return tf.gather(flat, index)

class SeqScorer(Scorer):

    def __init__(self, name):
        super(SeqScorer, self).__init__(name)
        self.lstm_nbr = 0

    def add_LSTM_input(self, input, hidden_size, num_layers = 1, num_proj = None, activation=tf.nn.tanh):
        scope = self.name + "_lstm" + str(self.lstm_nbr)
        self.lstm_nbr += 1
        with tf.variable_scope(scope):
            lstms = [tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(num_layers-1)]
            if num_proj is None:
                lstms.append(tf.contrib.rnn.LSTMCell(hidden_size))
            else:
                lstms.append(tf.contrib.rnn.LSTMCell(hidden_size, num_proj=num_proj))
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
            layer,_ = tf.nn.dynamic_rnn(multi_rnn_cell, input, dtype=tf.float32)
            lstm_output = transform_output(layer)
            self.add_input(lstm_output)

        return self
