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

class Filter(object):
    """Filters a sequence based on a condition.
    By default, the condition is the mean

    Parameters
    ----------
    seq : tf.placeholder or a tensor
        A sequence of shape [None, None, 1].
        The input must be sorted (descending)
    name : string
        The name of this block.

    Attributes
    ----------
    scope : type
        Description of attribute `scope`.
    seq

    """

    def __init__(self, seq, name):
        with tf.name_scope(name) as self.scope:
            self.seq = seq
            sh = tf.shape(seq)
            y = tf.reshape(seq, [sh[0], sh[1]])
            z = tf.zeros(tf.shape(y))
            self.threshold = self.generate_threshold()
            y = tf.where(y >= self.threshold, y, z)
            nz = tf.reduce_max(tf.count_nonzero(y, 1))
            y = y[:,:nz]
            self.graph = y

    def generate_threshold(self):
        return tf.reduce_mean(self.seq, 1)

    def get_threshold(self):
        return self.threshold

    def get_graph(self):
        return self.graph
