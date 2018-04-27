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
        vec = zip(*[vec*1]) # delete 1s
        return vec

    def get_doc_sim_vector(self, doc):
        sims = self.get_vector(doc + "/docInfo.json", "sims")
        extend(sims, self.get_docs_max_sim_length())
        return sims

    def get_doc_tf_vector(self, doc):
        freqs = self.get_vector(doc + "/docInfo.json", "freqs")
        extend(freqs, self.get_docs_max_tf_length())
        return freqs

    def get_doc_sizes_vector(self, doc):
        sizes = self.get_vector(doc + "/docInfo.json", "sizes")
        extend(sizes, self.get_docs_max_sizes_length())
        return sizes

    def get_sents_tf_vector(self, doc):
        freqs = self.get_vector(doc + "/sentTF.json", "freqs")
        extend(freqs, self.get_docs_max_tf_length())
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
