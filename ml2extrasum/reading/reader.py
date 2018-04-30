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

def extend(vector, max_len):
    diff = max_len - len(vector)
    if diff <= 0:
        return
    for _ in range(diff):
        vector.append([0.0])

def append_list(parent, child):
    if parent == None:
        parent = [child]
    else:
        parent.append(child)

class Reader(object):

    def __init__(self, url):
        self.url = url

    def set_lang(self, lang):
        self.lang_url = os.path.join(self.url, lang)

    def get_file_url(self, fname):
        return os.path.join(self.lang_url, fname)

    def get_max_length(self, property):
        json_url = self.get_file_url("lang.json")
        lang_properties = json.load(open(json_url))
        return int(lang_properties[property])

    def get_docs_max_sim_length(self):
        return self.get_max_length("maxDocSimLength")

    def get_docs_max_tf_length(self):
        return self.get_max_length("maxDocTFLength")

    def get_docs_max_sizes_length(self):
        return self.get_max_length("maxDocSizesLength")

    def get_sents_max_tf_length(self):
        return self.get_max_length("maxSentTFLength")

    def get_property(self, file, property):
        json_url = self.get_file_url(file)
        cont = json.load(open(json_url))
        return cont[property]

    def get_vector(self, file, property):
        vec = self.get_property(file, property)
        vec.sort(reverse=True)
        vec = zip(*[vec*1])
        return vec

    def get_doc_sim_vector(self, doc):
        sims = self.get_vector(doc + "/docInfo.json", "sims")
        #extend(sims, self.get_docs_max_sim_length())
        return sims

    def get_doc_tf_vector(self, doc):
        freqs = self.get_vector(doc + "/docInfo.json", "freqs")
        #extend(freqs, self.get_docs_max_tf_length())
        return freqs

    def get_doc_sizes_vector(self, doc):
        sizes = self.get_vector(doc + "/docInfo.json", "sizes")
        #extend(sizes, self.get_docs_max_sizes_length())
        return sizes

    def get_sents_tf_vector(self, doc):
        freqs = self.get_vector(doc + "/sentTF.json", "freqs")
        #extend(freqs, self.get_docs_max_tf_length())
        return freqs

    def get_doc_size(self, doc):
        return [self.get_property(doc + "/docInfo.json", "size")]

    def create_lang_list(self, func):
        lang_list = None
        for f in os.listdir(self.lang_url):
            d = os.path.join(self.lang_url, f)
            if os.path.isdir(d):
                vec = func(d)
                if lang_list == None:
                    lang_list = [vec]
                else:
                    lang_list.append(vec)
        return lang_list

    def get_doc_tf_lang(self):
        return self.create_lang_list(self.get_doc_tf_vector)

    def get_doc_sim_lang(self):
        return self.create_lang_list(self.get_doc_sim_vector)

    def get_doc_sizes_lang(self):
        return self.create_lang_list(self.get_doc_sizes_vector)

    def get_sents_tf_lang(self):
        return self.create_lang_list(self.get_sents_tf_vector)

    def get_doc_size_lang (self):
        return self.create_lang_list(self.get_doc_size)

    def process_sents_in_doc(self, doc_path, tf_vec, sim_vec, size_vec, pos_vec, rouge_vec):
        tf_cont = json.load(open(doc_path + "/sentTF.json"))
        sim_cont = json.load(open(doc_path + "/sentSim.json"))
        size_cont = json.load(open(doc_path + "/sentLen.json"))
        rouge1_cont = json.load(open(doc_path + "/sentRouge1.json"))

        for num in sorted(tf_cont):
            v = tf_cont[num]
            v.sort(reverse=True)
            tf_vec.append(zip(*[v*1]))
            v = sim_cont[num]
            v.sort(reverse=True)
            sim_vec.append(zip(*[v*1]))

            size_vec.append([size_cont[num]])
            pos_vec.append([num])
            rouge_vec.append([rouge1_cont[num]])

        return len(tf_cont)

    def create_doc_batch(self):
        doc_batch = {}
        for f in os.listdir(self.lang_url):
            d = os.path.join(self.lang_url, f)
            if os.path.isdir(d):
                doc_batch[f] = {}
                doc_batch[f]["doc_tf_seq"] = self.get_doc_tf_vector(d)
                doc_batch[f]["doc_sim_seq"] = self.get_doc_sim_vector(d)
                doc_batch[f]["doc_size_seq"] = self.get_doc_sizes_vector(d)

                tf_vec = []
                sim_vec = []
                size_vec = []
                pos_vec = []
                rouge_vec = []
                nbr_sents = self.process_sents_in_doc(d, tf_vec, sim_vec, size_vec, pos_vec, rouge_vec)

                doc_batch[f]["nbr_sents"] = nbr_sents
                doc_batch[f]["sent_tf_seq"] = tf_vec
                doc_batch[f]["sent_sim_seq"] = sim_vec
                doc_batch[f]["sent_size"] = sim_vec
                doc_batch[f]["sent_pos"] = pos_vec
                doc_batch[f]["rouge_1"] = rouge_vec

        return doc_batch
