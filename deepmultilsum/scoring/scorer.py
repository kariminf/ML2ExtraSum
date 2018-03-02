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

    def add_input(self, input):
        if self.net is None :
            self.net = input
        else:
            self.net = tf.concat((self.net, input), axis=1)
        return self

    def add_layer(self, units, activation=tf.nn.relu):

        if self.net == None:
            return self

        self.layers += 1
        self.net = tf.layers.dense(self.net, units=units, activation=activation)
        return self

    def get_output(self):
        return self.net
