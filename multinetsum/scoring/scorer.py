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

class Scorer(object):

    def __init__(self, name):
        self.name = name
        self.input = None
        self.layers = []
        self.output = [[]]
        self.inputOn = True

    def add_input(self, input):
        if self.inputOn:
            if self.input is None :
                self.input = input
            else:
                self.input = tf.concat((self.input, input), axis=1)
        return self

    def add_layer(self, nbr_noads):
        layer_nbr = len(self.layers)
        theta_name = self.name + "_theta" + str(layer_nbr)
        bias_name = self.name + "_bias" + str(layer_nbr)

        layer_input = self.input

        if layer_nbr > 0:
            layer_input = self.layers[-1]

        nbr_inputs = int(layer_input.get_shape()[1])
        #print nbr_inputs
        theta = tf.Variable(tf.random_uniform([nbr_inputs,nbr_noads], -1, 1), name=theta_name)
        bias = tf.Variable(tf.zeros([nbr_noads]), name=bias_name)
        layer = tf.tanh(tf.matmul(layer_input, theta) + bias)
        self.layers.append(layer)

        return self

    def create(self, nbr_outputs):
        self.add_layer(nbr_outputs)
        self.output = self.layers[-1]
        return self

    def get_output(self):
        return self.output
