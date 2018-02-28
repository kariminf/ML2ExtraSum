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
from multinetsum.scoring.scorer import Scorer

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

    def add_LSTM_input(self, input, nbr_noads, nbr_outputs):

        if not self.inputOn:
            return self

        scope = self.name + "_lstm" + str(self.lstm_nbr)
        self.lstm_nbr += 1
        with tf.variable_scope(scope):
            lstm = tf.contrib.rnn.LSTMCell(nbr_noads ,num_proj=nbr_outputs)
            layer,_ = tf.nn.dynamic_rnn(lstm, input, dtype=tf.float32)
            lstm_output = transform_output(layer)
            if self.input is None:
                self.input = lstm_output
            else:
                self.input = tf.concat((self.input, tstm_output), axis=1)

        return self
