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

#just a code to learn how to sum a vector then divide it on the sum of another
#victor, then multiply by a scalar

import tensorflow as tf

#import os
import sys
sys.path.insert(0, "..")
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml2extrasum.filtering.filter import Filter

import numpy as np

X = [
	[[15.0], [20.0], [0.0], ],
	[[10.0], [12.0], [0.0]],
	[[10.0], [12.0], [0.0]],
	[[2.0], [12.0], [13.0]],
	[[5.0], [0.0], [0.0]]
]


if __name__ == '__main__':
    x_ = tf.placeholder(tf.float32, shape=[None,None,1], name="x-input")
	
    graph = SeqScorer("percentage")

    graph.add_LSTM_input(x_, 2, 2).add_LSTM_input(y_, 2, 2) #lstm1, lstm2

    graph.add_input(z_)

    graph.add_layer(10, tf.nn.tanh).add_layer(1, tf.nn.tanh)

    output = graph.get_output()

    cost = tf.losses.mean_squared_error(r_, output)

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    for i in range(10000):
        _, cst = sess.run([train_step, cost], feed_dict={x_: X, y_: Y, z_: Z , r_: RESULT})
        print i, cst

	#saver.save(sess, "./aak.ckpt")
    tX = [[[10.0], [5.0]]]
    tY = [[[10.0], [5.0], [15.0], [8.0]]]
    tZ = [[0.8]]
    tRES = [[0.3157894737]]

    print("Predicted: ", sess.run(output,feed_dict={x_: tX, y_: tY, z_: tZ}), " Expected: ", tRES)
