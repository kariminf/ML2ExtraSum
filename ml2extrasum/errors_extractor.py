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

STATS_DIR = "/home/kariminf/Data/ATS/Mss15Test/stats/"

def extract_vec(cont):
    rouge_vec = []
    for num in sorted(cont):
        rouge_vec.append([float(cont[num])])
    return rouge_vec

# Data reading
# ===============

csv = "lang, min, mean, max\n"

path = os.path.join("./outputs/", "test.json")

cont = json.load(open(path))

#I store the errors just in case I use them ulteriorly
errors = {}

for lang in cont:
    lang_data = cont[lang]
    errors[lang] = []
    print lang
    for doc in lang_data:
        rouge1_path = os.path.join(STATS_DIR, lang, doc, "sentRouge1.json")
        rouge1_expected = extract_vec(json.load(open(rouge1_path)))
        rouge1_infered = cont[lang][doc]["sent"]
        # mean squared error
        error = np.square(np.subtract(rouge1_expected, rouge1_infered)).mean()
        errors[lang].append(error)
    csv += lang + "," + str(np.min(errors[lang])) + "," + str(np.mean(errors[lang])) + "," + str(np.max(errors[lang])) + "\n"


with open("outputs/errors.csv", "w") as text_file:
    text_file.write(csv)

"""
with open('data.json', 'w') as fp:
    json.dump(scores, fp)
"""
