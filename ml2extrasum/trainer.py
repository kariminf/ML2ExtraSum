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
from reading.reader import Reader


def repeat_vector(vector, nbr):
    return [vector] * nbr

STATS_DIR = "/home/kariminf/Data/ATS/Mss15Train/stats/"
TRAIN_ITER = 2

#Inputs holders
#===============
# term frequencies in document
doc_tf_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_tf_seq_in")
# all sentences similarities in a document
doc_sim_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_sim_seq_in")
# all sentences sizes in a document
doc_size_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_size_seq_in")
# document size
doc_size_ = tf.placeholder(tf.float32, shape=[None,1], name="doc_size_in")
# term frequencies (in the document) of a sentence
sent_tf_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_tf_seq_in")
# similarities of this sentence with others
sent_sim_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="sent_sim_seq_in")
# sentence size
sent_size_ = tf.placeholder(tf.float32, shape=[None,1], name="sent_size_in")
# sentence position
sent_pos_ = tf.placeholder(tf.float32, shape=[None,1], name="sent_pos_in")

rouge_1_ = tf.placeholder(tf.float32, shape=[None,1], name="rouge_1_out")

model = StatNet(doc_tf_seq_, doc_sim_seq_, doc_size_seq_, doc_size_, \
                sent_tf_seq_, sent_sim_seq_, sent_size_, sent_pos_)

output = model.get_graph()

cost = tf.losses.mean_squared_error(rouge_1_, output)

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

data = {}

reader = Reader(STATS_DIR)

for lang in os.listdir(STATS_DIR):
    lang_url = os.path.join(STATS_DIR, lang)
    if os.path.isdir(lang_url):
        print "reading ", lang
        reader.set_lang(lang)
        data[lang] = reader.create_doc_batch()

for i in range(TRAIN_ITER):
    for lang in data:
        lang_data = data[lang]
        for doc in lang_data:
            doc_data = lang_data[doc]
            nbr_sents = doc_data["nbr_sents"]
            feed = {

            doc_tf_seq_ : repeat_vector(doc_data["doc_tf_seq"], nbr_sents),
            doc_sim_seq_ : repeat_vector(doc_data["doc_sim_seq"], nbr_sents),
            doc_size_seq_ : repeat_vector(doc_data["doc_size_seq"], nbr_sents),
            doc_size_ : repeat_vector([nbr_sents], nbr_sents),

            sent_tf_seq_ : doc_data["sent_tf_seq"],
            sent_sim_seq_ : doc_data["sent_sim_seq"],
            sent_size_ : doc_data["sent_size"],
            sent_pos_ : doc_data["sent_pos"],
            rouge_1_ : doc_data["rouge_1"]

            }
            _, cst = sess.run([train_step, cost], feed_dict=feed)
            print i, cst
