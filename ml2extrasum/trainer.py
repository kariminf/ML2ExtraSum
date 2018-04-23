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

import os
import tensorflow as tf
from modeling.stat_net import StatNet
from reading import reader

#

STATS_DIR = "/home/kariminf/Data/ATS/Mss15Train/stats/"
TRAIN_ITER = 2

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

saver = tf.train.Saver()

batch = {}

for f in os.listdir(dataset_url):
    lang_url = os.path.join(dataset_url, f)
    if os.path.isdir(lang_url):
        print "reading ", f
        batch[f]["doc_tf_seq"] = reader.get_doc_tf_lang(lang_dir)
        batch[f]["doc_sim_seq"] = reader.get_doc_sim_lang(lang_dir)
        batch[f]["doc_size_seq"] = reader.get_doc_sizes_lang(lang_dir)
        batch[f]["doc_size"] = reader.get_doc_size_lang(lang_dir)

        batch[f]["sent_tf_seq"] = reader.get_sent_tf_lang(lang_dir)
        
        sent_sim_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_sim_seq_in")
        sent_size_ = tf.placeholder(tf.float32, shape=[None,1], name="sent_size_in")
        sent_pos_ = tf.placeholder(tf.float32, shape=[None,1], name="sent_pos_in")

        doc_sim_seq = reader.get_doc_sim_lang(lang_url)
        batch[f] = doc_sim_seq

for i in range(TRAIN_ITER):
    for lang_dir in os.listdir(STATS_DIR):
        lang_dir = os.path.join(STATS_DIR, lang_dir)
        if os.path.isdir(lang_dir):


    _, cst = sess.run([train_step, cost], feed_dict={x_: X, y_: Y, z_: Z , r_: RESULT})
    print i, cst
