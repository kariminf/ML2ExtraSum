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

import sys
import numpy
import tensorflow as tf

sys.path.insert(0, "..")

from ml2extrasum.modeling.seq_autoencoder import SeqAutoEncoder

S = [
[[0.2], [0.6], [0.8], [0.9]],
[[0.1], [0.3], [0.4], [0.45]],
[[0.7], [0.8], [0.9], [0.0]]
]

LEARNING_RATE = 0.005
max_grad_norm = 5

#Inputs holders
seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="seq")

model = SeqAutoEncoder("seq_test", seq_)

latent = model.get_latent()


# Train the point in latent space to have zero-mean and unit-variance on batch basis
lat_mean, lat_var = tf.nn.moments(latent, axes=[1])
loss_lat_batch = tf.reduce_mean(tf.square(lat_mean) + lat_var - tf.log(lat_var) - 1)


output = model.get_graph()

output2 = tf.squeeze(seq_)

"""
h_sigma = tf.exp(output)
dist = tf.contrib.distributions.Normal(output, output)
px = dist.log_prob(tf.transpose(output2))
loss_seq = -px
loss_seq = tf.reduce_mean(loss_seq)
"""

"""
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.1, staircase=False)
loss = loss_seq + loss_lat_batch

# Route the gradients so that we can plot them on Tensorboard
tvars = tf.trainable_variables()

grads = tf.gradients(loss, tvars)
grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

optimizer = tf.train.AdamOptimizer(lr)
gradients = zip(grads, tvars)
train_step = optimizer.apply_gradients(gradients, global_step=global_step)
"""

loss = tf.losses.mean_squared_error(output2, output) + loss_lat_batch

train_step = tf.train.AdamOptimizer().minimize(loss)

#cost = tf.losses.mean_squared_error(output2, output)

#train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Sshape = tf.shape(S)
#I = sess.run(tf.reshape(S, [Sshape[0], Sshape[1]]))
#I = sess.run(tf.squeeze(S))

#i = sess.run(tf.reshape(S, [3, 4]))

for i in range(10000):
    _, cst = sess.run([train_step, loss], feed_dict={seq_: S})
    print i, cst

o, l = sess.run([output, latent], feed_dict={seq_: [S[0]]})

print "ideal output", S[0]
print "infered output", o
print "latent:", l[-1]

o, l = sess.run([output, latent], feed_dict={seq_: [S[1]]})

print "ideal output", S[1]
print "infered output", o
print "latent:", l[-1]

o, l = sess.run([output, latent], feed_dict={seq_: [S[2]]})

print "ideal output", S[2]
print "infered output", o
print "latent:", l[-1]

o, l = sess.run([output, latent], feed_dict={seq_: [[ [0.5], [0.3], [0.3], [0.3], [0.3], [0.3] ]]})

print "ideal output", [[ [0.5], [0.3], [0.3], [0.3], [0.3], [0.3] ]]
print "infered output", o
print "latent:", l[-1]
