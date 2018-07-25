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
from modeling.stat_net import StatNet

#from reading.reader import Reader
from reading.limited_reader import LimitedReader


STATS_DIR = "/home/kariminf/Data/ATS/Mss15Train/stats0+/"
MODEL_DIR = "/home/kariminf/Data/ATS/Models/en_ar_100it/stat0Model.ckpt"

def repeat_vector(vector, nbr):
    return [vector] * nbr

model = StatNet()

model.restore("models/stat0m")

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
# ===============sent.tolist()

sqr = ""

for lang in data:
    lang_data = data[lang]
    lang_scores = {}
    sqr += "=== " + lang + " ===\n"
    for doc in lang_data:

        print doc
        sqr += "=== " + doc + " ===\n"

        doc_data = lang_data[doc]

        scores = model.test(doc_data)

        #sqr += "cost: " + str(scores["cost"]) + "\n"
        sqr += "lang: " + str(scores["lang"]) + "\n"
        #sqr += "tf: " + str(scores["tf"]) + "\n"
        #sqr += "sim: " + str(scores["sim"]) + "\n"
        #sqr += "pos: " + str(scores["pos"]) + "\n"
        #sqr += "sent: " + str(scores["sent"]) + "\n"


    #scores[lang] = lang_scores
with open("Output.txt", "w") as text_file:
    text_file.write(sqr)

"""
with open('data.json', 'w') as fp:
    json.dump(scores, fp)
"""
