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

#just a code to learn and test (a ==> b)

import tensorflow as tf

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multinetsum.scoring.scorer import Scorer

X = [
[1.0, 1.0],
[0.0, 0.0],
[1.0, 0.0],
[0.0, 1.0]
]

R = [
[1.0],
[1.0],
[0.0],
[1.0]
]

LEARNING_RATE = 0.05

x = tf.placeholder(tf.float32, shape=[None, 2], name="inputs")
r = tf.placeholder(tf.float32, shape=[None ,1], name="result")

s1 = Scorer("scorer1").add_input(x).add_layer(4).create(1)

output = s1.get_output()

output = tf.add(output, 1)
output = tf.multiply(output, 0.5)

cost = - tf.reduce_mean( (r * tf.log(output)) + (1 - r) * tf.log(1.0 - output)  )
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(20000):
    _, cst = sess.run([train_step, cost], feed_dict={x: X, r: R})
    print i, cst

tX = [[1.0, 0.4]]
tR = [[0.0]]
print("Predicted: ", sess.run(output,feed_dict={x: tX}), " Expected: ", tR)
