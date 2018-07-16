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
from modeling.stat_net import StatNet
#from reading.reader import Reader
from reading.limited_reader import LimitedReader


def repeat_vector(vector, nbr):
    return [vector] * nbr

STATS_DIR = "/home/kariminf/Data/ATS/Mss15Train/stats0/"
TRAIN_ITER = 20
LEARNING_RATE = 0.05

model = StatNet()

data = {}

reader = LimitedReader(STATS_DIR)

lang_url = os.path.join(STATS_DIR, "en")
if os.path.isdir(lang_url):
    print "reading "
    reader.set_lang("en")
    data["en"] = reader.create_doc_batch()

sess = model.get_session()
writer = tf.summary.FileWriter("models", sess.graph)
lang_data = data["en"]
for doc in lang_data:
    doc_data = lang_data[doc]
    cst = model.train(doc_data)
    print doc, " ==> cost: ", cst

writer.close()
