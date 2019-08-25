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

import os, sys
import json
import numpy as np
import tensorflow as tf

#from reading.reader import Reader
from reading.limited_reader import LimitedReader

import utils

config = utils.get_config()

def repeat_vector(vector, nbr):
    return [vector] * nbr

MODEL = utils.get_model(config["StatNet"])

model = MODEL()

model.restore(config["MOD_DIR"] + config["MOD_NAME"])

# Data reading
# ===============
data = {}

reader = LimitedReader(config["TEST_DIR"])

json = "{"

lang_next = False

for lang in os.listdir(config["TEST_DIR"]):
    lang_url = os.path.join(config["TEST_DIR"], lang)
    if os.path.isdir(lang_url):
        print "reading ", lang
        reader.set_lang(lang)
        lang_data = reader.create_doc_batch()
        if lang_next:
            json += "\t},\n"
        else:
            lang_next = True
        lang_scores = {}
        json += "\t\"" + lang + "\": {\n"
        doc_next = False
        for doc in lang_data:
            if doc_next:
                json += "\t\t},\n"
            else:
                doc_next = True
            print doc
            json += "\t\t\"" + doc + "\":{\n"

            doc_data = lang_data[doc]

            scores = model.test(doc_data)

            json += "\t\t\t\"cost\": " + str(scores["cost"]) + ",\n"
            json += "\t\t\t\"lang\": " + str(scores["lang"]) + ",\n"
            json += "\t\t\t\"tf\": " + str(scores["tf"]) + ",\n"
            json += "\t\t\t\"sim\": " + str(scores["sim"]) + ",\n"
            json += "\t\t\t\"pos\": " + str(scores["pos"]) + ",\n"
            json += "\t\t\t\"size\": " + str(scores["size"]) + ",\n"
            json += "\t\t\t\"sent\": " + str(scores["sent"]) + "\n"
        # The last doc
        json += "\t\t}\n"

# The last lang
json += "\t}\n"
# The global block
json += "}\n"

    #scores[lang] = lang_scores
with open(config["SUM_DIR"] + "scores.json", "w") as text_file:
    text_file.write(json)

"""
with open('data.json', 'w') as fp:
    json.dump(scores, fp)
"""
