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
from modeling.stat_net import StatNet

#

#Inputs holders
doc_tf_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_tf_seq_in")
doc_sim_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_sim_seq_in")
doc_size_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_size_seq_in")
doc_size_ = tf.placeholder(tf.float32, shape=[None,1], name="doc_size_in")
sent_tf_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_tf_seq_in")
sent_sim_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_sim_seq_in")
sent_size_ = tf.placeholder(tf.float32, shape=[None,1], name="sent_size_in")
sent_pos_ = tf.placeholder(tf.float32, shape=[None,1], name="sent_pos_in")

model = StatNet(doc_tf_seq_, doc_sim_seq_, doc_size_seq_, doc_size_, \
                sent_tf_seq_, sent_sim_seq_, sent_size_, sent_pos_)

output = model.get_graph()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
