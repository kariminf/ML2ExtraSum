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
import numpy

import tensorflow as tf

from modeling.seq_autoencoder import SeqAutoEncoder
from reading import reader

dataset_url = "/home/kariminf/Data/ATS/Mss15Train/stats/"

batch = {}

for f in os.listdir(dataset_url):
    lang_url = os.path.join(dataset_url, f)
    if os.path.isdir(lang_url):
        print "reading ", f
        doc_sim_seq = reader.get_doc_sim_lang(lang_url)
        batch[f] = doc_sim_seq

#Inputs holders
doc_sim_seq_ = tf.placeholder(tf.float32, shape=[None,None,1], name="doc_sim_seq_in")

model = SeqAutoEncoder("doc_sim", doc_sim_seq_)

latent = model.get_latent()

output = model.get_graph()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100):
    for lang in batch:
        doc_sim_seq = batch[lang]
        print i, lang, numpy.shape(doc_sim_seq)
        #_, cst, o = sess.run([train_step, cost, out], feed_dict={x: X, r: R})
