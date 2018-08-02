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
from extracting.sim_extractor import SimExtractor

STATS_DIR = "/home/kariminf/Data/ATS/Mss15Test/stats/"
SIZE_DIR = "/home/kariminf/Data/ATS/Mss15Test/src/target-length/"
DEST_DIR = "/home/kariminf/Data/ATS/Mss15Test/tests/testing2018/"

def get_sentences(lang, doc):
    path = os.path.join(STATS_DIR, lang, doc, "sents.json")
    cont = json.load(open(path))
    path = os.path.join(STATS_DIR, lang, doc, "sentRouge1.json")
    rouge1_cont = json.load(open(path))

    sentences = []
    sent_ids = []
    rouge1 = []
    for num in sorted(cont):
        sentences.append(cont[num])
        rouge1.append(rouge1_cont[num])
        sent_ids.append(num)
    path = os.path.join(STATS_DIR, lang, doc, "sentSimZ.json")
    sims = json.load(open(path))
    return sent_ids, sentences, sims, rouge1

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

def write_summary(extractor, size, lang, doc, name):
    sents = extractor.extract_n_chars(size)
    summary = "\n".join(sents)
    path = os.path.join(DEST_DIR, lang, doc)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, name + ".txt")
    with codecs.open(path, "w", "utf8") as text_file:
        text_file.write(summary)

path = os.path.join("./outputs/", "test.json")

cont = json.load(open(path))

for lang in cont:
    lang_data = cont[lang]
    sizes = get_summary_sizes(lang)
    print lang
    for doc in lang_data:
        doc_data = cont[lang][doc]
        size = sizes[doc + "_body.txt"]
        ids, sentences, sims, doc_data["rouge1"] = get_sentences(lang, doc)
        doc_data["sums"] = [x + y + z + w for x, y, z, w in
            zip(doc_data["tf"], doc_data["sim"], doc_data["pos"], doc_data["size"])]
        #scores
        for score_name in ["sent", "tf", "sim", "pos", "size", "rouge1", "sums"]:
            extractor = Extractor(sentences, doc_data[score_name])
            sim_extractor = SimExtractor(sentences, doc_data[score_name], ids, sims)
            write_summary(extractor, size, lang, doc, score_name + "_plain")
            write_summary(sim_extractor, size, lang, doc, score_name + "_sim")
