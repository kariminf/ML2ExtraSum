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
        self.net = None
        self.layers = 0
        self.done = False
        self.inputs = []
        with tf.name_scope(name) as self.scope:
            pass

    def add_input(self, input):
        if self.done:
            return self
        if self.net is None :
            self.net = input
        """
        if self.net is None :
            self.net = input
        else:
            self.inputs.append(input)
        """
        self.inputs.append(input)
        return self

    def add_hidden(self, units, activation=tf.nn.relu, name=None):
        if self.done or self.net == None:
            return self

        if self.layers == 0:
            with tf.name_scope(self.scope):
                self.net = tf.concat(self.inputs, axis=1, name=self.name + "-in")

        self.layers += 1
        """
        if name == None:
            name = self.name + str(self.layers)
        """
        with tf.name_scope(self.scope):
            self.net = tf.layers.dense(self.net, units=units, activation=activation)
        return self

    def add_output(self, units, activation=tf.nn.relu):
        if not self.done:
            self.add_hidden(units, activation, self.name)
        self.done = True
        return self.net

    def get_output(self):
        if not self.done:
            return None
        return self.net
