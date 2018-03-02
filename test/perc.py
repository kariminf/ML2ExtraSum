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
import numpy as np

DATA = [
	[[15, 20], [15, 20, 30], 0.5],
	[[10, 12], [15, 12, 12, 10], .3],
	[[10, 12], [15, 12, 12, 10], .2],
	[[2, 12, 13], [20, 13, 2, 12, 25], .1],
	[[5], [5, 13, 12, 25], .7]
]

X = [
	[[15.0], [20.0], [0.0]],
	[[10.0], [12.0], [0.0]],
	[[10.0], [12.0], [0.0]],
	[[2.0], [12.0], [13.0]],
	[[5.0], [0.0], [0.0]]
]

X__ = [
	[[0.0], [15.0], [20.0]],
	[[0.0], [10.0], [12.0]],
	[[0.0], [10.0], [12.0]],
	[[2.0], [12.0], [13.0]],
	[[0.0], [0.0], [5.0]]
]

Y = [
	[[15.0], [20.0], [30.0], [0.0], [0.0]],
	[[15.0], [12.0], [12.0], [10.0], [0.0]],
	[[15.0], [12.0], [12.0], [10.0], [0.0]],
	[[20.0], [13.0], [2.0], [12.0], [25.0]],
	[[5.0], [13.0], [12.0], [25.0], [0.0]]
]

Y__ = [
	[[0.0], [0.0], [15.0], [20.0], [30.0]],
	[[0.0], [15.0], [12.0], [12.0], [10.0]],
	[[0.0], [15.0], [12.0], [12.0], [10.0]],
	[[20.0], [13.0], [2.0], [12.0], [25.0]],
	[[0.0], [5.0], [13.0], [12.0], [25.0]]
]

Z = [
	[0.5],
	[0.3],
	[0.2],
	[0.1],
	[0.7]
]

RESULT = [
	[0.2692307692],
	[0.1346938776],
	[0.0897959184],
	[0.0375],
	[0.0636363636]
]

LEARNING_RATE = 0.05

def get_output(net):
	batch_size = tf.shape(net)[0]
	max_length = tf.shape(net)[1]
	out_size = int(net.get_shape()[2])
	index = tf.range(0, batch_size) * max_length
	flat = tf.reshape(net, [-1, out_size])
	return tf.gather(flat, index)

if __name__ == '__main__':

	##############################################################################

	x_ = tf.placeholder(tf.float32, shape=[None,None,1], name="x-input")
	y_ = tf.placeholder(tf.float32, shape=[None,None,1], name="y-input")
	z_ = tf.placeholder(tf.float32, shape=[None,1], name="z-input")

	r_ = tf.placeholder(tf.float32, shape=[None ,1], name="result")

	with tf.variable_scope("lstm1"):
		sumXCell = tf.contrib.rnn.LSTMCell(2,num_proj=2)
		outX,_ = tf.nn.dynamic_rnn(sumXCell,x_,dtype=tf.float32)   #shape: (None, 12, 2)

	with tf.variable_scope("lstm2"):
		sumYCell = tf.contrib.rnn.LSTMCell(2, num_proj=2)
		outY,_ = tf.nn.dynamic_rnn(sumYCell,y_,dtype=tf.float32)   #shape: (None, 12, 2)

	predXS = tf.reduce_sum(outX)

	outXY = tf.concat((get_output(outX), get_output(outY), z_), axis=1)

	output = tf.layers.dense(outXY, units=10, activation=tf.nn.tanh)

	output = tf.layers.dense(output, units=1, activation=tf.nn.tanh)

	cost = tf.losses.mean_squared_error(r_, output)

	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	#saver = tf.train.Saver()

	for i in range(10000):
		_, cst, px = sess.run([train_step, cost, predXS], feed_dict={x_: X, y_: Y, z_: Z , r_: RESULT})
		print i, cst, px
		#sess.run(train_step, feed_dict={x_: X, y_: Y, z_: Z, r_: RESULT})

	#saver.save(sess, "./aak.ckpt")
	tX = [[[10.0], [5.0]]]
	tY = [[[10.0], [5.0], [15.0], [8.0]]]
	tZ = [[0.8]]
	tRES = [[0.3157894737]]

	print("Predicted: ", sess.run(output,feed_dict={x_: tX, y_: tY, z_: tZ}), " Expected: ", tRES)
