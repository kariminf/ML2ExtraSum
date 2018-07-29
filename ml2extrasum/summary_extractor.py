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
import codecs
from extracting.extractor import Extractor

STATS_DIR = "/home/kariminf/Data/ATS/Mss15Test/stats/"
SIZE_DIR = "/home/kariminf/Data/ATS/Mss15Test/src/target-length/"
DEST_DIR = "/home/kariminf/Data/ATS/Mss15Test/tests/testing2018/"

def get_sentences(lang, doc):
    path = os.path.join(STATS_DIR, lang, doc, "sents.json")
    cont = json.load(open(path))
    sentences = []
    for num in sorted(cont):
        sentences.append(cont[num])
    return sentences

def get_summary_sizes(lang):
    path = os.path.join(SIZE_DIR, lang + ".txt")
    sizes = {}
    file = open(path, 'r')
    while 1:
        line = file.readline()
        if line == '':
			break;
        parts = line.split(",")
        sizes[parts[0]] = int(parts[1])
    file.close()
    return sizes

def write_summary(summary, lang, doc):
    path = os.path.join(DEST_DIR, lang)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, doc + ".txt")
    with codecs.open(path, "w", "utf8") as text_file:
        text_file.write(summary)

path = os.path.join("./outputs/", "test.json")

cont = json.load(open(path))

for lang in cont:
    lang_data = cont[lang]
    sizes = get_summary_sizes(lang)
    print lang
    for doc in lang_data:
        rouge1_scores = cont[lang][doc]["sent"]
        sentences = get_sentences(lang, doc)
        extractor = Extractor(sentences, rouge1_scores)
        sents = extractor.extract_n_chars(sizes[doc + "_body.txt"])
        summary = "\n".join(sents)
        write_summary(summary, lang, doc)
