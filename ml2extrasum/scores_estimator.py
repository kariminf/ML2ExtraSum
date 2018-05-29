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
import json
import numpy as np
import tensorflow as tf

#from reading.reader import Reader
from reading.limited_reader import LimitedReader


STATS_DIR = "/home/kariminf/Data/ATS/Mss15Train/stats0/"
MODEL_DIR = "/home/kariminf/Data/ATS/Models/en_ar_100it/stat0Model.ckpt"

def repeat_vector(vector, nbr):
    return [vector] * nbr

# Restoring the trained model
# ============================
sess = tf.Session()

model = tf.train.import_meta_graph(MODEL_DIR + '.meta')

model.restore(sess, MODEL_DIR)

graph = tf.get_default_graph()


#Inputs holders
#===============
# term frequencies in document
doc_tf_seq_ = graph.get_tensor_by_name("doc_tf_seq_in:0")
# all sentences similarities in a document
doc_sim_seq_ = graph.get_tensor_by_name("doc_sim_seq_in:0")
# all sentences sizes in a document
doc_size_seq_ = graph.get_tensor_by_name("doc_size_seq_in:0")
# document size
doc_size_ = graph.get_tensor_by_name("doc_size_in:0")
# term frequencies (in the document) of a sentence
sent_tf_seq_ = graph.get_tensor_by_name("sent_tf_seq_in:0")
# similarities of this sentence with others
sent_sim_seq_ = graph.get_tensor_by_name("sent_sim_seq_in:0")
# sentence size
sent_size_ = graph.get_tensor_by_name("sent_size_in:0")
# sentence position
sent_pos_ = graph.get_tensor_by_name("sent_pos_in:0")

#rouge_1_ = tf.placeholder(tf.float32, shape=[None,1], name="rouge_1_out")

# Restoring the scorers
# =======================

lang_scorer = graph.get_tensor_by_name("lang_scorer/kernel:0")

tf_scorer = graph.get_tensor_by_name("tf_scorer/kernel:0")
sim_scorer = graph.get_tensor_by_name("sim_scorer/kernel:0")
size_scorer = graph.get_tensor_by_name("size_scorer/kernel:0")
pos_scorer = graph.get_tensor_by_name("pos_scorer/kernel:0")

sent_scorer = graph.get_tensor_by_name("sent_scorer/kernel:0")


# Data reading
# ===============
data = {}

reader = LimitedReader(STATS_DIR)

for lang in os.listdir(STATS_DIR):
    lang_url = os.path.join(STATS_DIR, lang)
    if os.path.isdir(lang_url):
        print "reading ", lang
        reader.set_lang(lang)
        data[lang] = reader.create_doc_batch()

# Scoring
# ===============

scores = {}

sqr = ""

for lang in data:
    lang_data = data[lang]
    lang_scores = {}
    sqr += "=== " + lang + " ===\n"
    for doc in lang_data:

        print doc
        sqr += "=== " + doc + " ===\n"

        doc_data = lang_data[doc]
        nbr_sents = doc_data["nbr_sents"]

        print "doc_tf_seq=", np.shape(repeat_vector(doc_data["doc_tf_seq"], nbr_sents))

        feed = {
        doc_tf_seq_ : repeat_vector(doc_data["doc_tf_seq"], nbr_sents),
        doc_sim_seq_ : repeat_vector(doc_data["doc_sim_seq"], nbr_sents),
        doc_size_seq_ : repeat_vector(doc_data["doc_size_seq"], nbr_sents),
        doc_size_ : repeat_vector([nbr_sents], nbr_sents),
        sent_tf_seq_ : doc_data["sent_tf_seq"],
        sent_sim_seq_ : doc_data["sent_sim_seq"],
        sent_size_ : doc_data["sent_size"],
        sent_pos_ : doc_data["sent_pos"]
        #rouge_1_ : doc_data["rouge_1"]
        }

        lang, tfreq, sim, size, pos, sent = sess.run([lang_scorer, tf_scorer, sim_scorer, size_scorer, pos_scorer, sent_scorer], feed_dict=feed)

        lang_scores[doc] = {}
        lang_scores[doc]["lang"] = lang.tolist()
        lang_scores[doc]["tf"] = tfreq.tolist()
        lang_scores[doc]["sim"] = sim.tolist()
        lang_scores[doc]["pos"] = pos.tolist()
        lang_scores[doc]["sent"] = sent.tolist()

        print np.shape(lang)
        print np.shape(sent)

        sqr += "lang: " + str(lang.tolist()) + "\n"
        sqr += "tf: " + str(tfreq.tolist()) + "\n"
        sqr += "sim: " + str(sim.tolist()) + "\n"
        sqr += "pos: " + str(pos.tolist()) + "\n"
        sqr += "sent: " + str(sent.tolist()) + "\n"


    #scores[lang] = lang_scores
with open("Output.txt", "w") as text_file:
    text_file.write(sqr)

"""
with open('data.json', 'w') as fp:
    json.dump(scores, fp)
"""
