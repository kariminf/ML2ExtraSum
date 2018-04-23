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

def get_docs_max_sim_length(lang_url):
    json_url = os.path.join(lang_url, "lang.json")
    lang_properties = json.load(open(json_url))
    return int(lang_properties["maxDocSimLength"])

def get_doc_sim_vector(doc_url):
    json_url = os.path.join(doc_url, "docInfo.json")
    doc_info = json.load(open(json_url))
    sims = doc_info["sims"]
    sims = zip(*[sims*1])
    return sims

def extend(vector, max_len):
    diff = max_len - len(vector)
    if diff <= 0:
        return
    for _ in range(diff):
        vector.append([0.0])


def get_doc_sim_lang (lang_url):
    max_len = get_docs_max_sim_length(lang_url)
    doc_sim_lang = None
    for f in os.listdir(lang_url):
        d = os.path.join(lang_url, f)
        if os.path.isdir(d):
            doc_sim = get_doc_sim_vector(d)
            extend(doc_sim, max_len)
            if doc_sim_lang == None:
                doc_sim_lang = [doc_sim]
            else:
                doc_sim_lang.append(doc_sim)
    return doc_sim_lang

# ====================================================

def get_docs_max_tf_length(lang_url):
    json_url = os.path.join(lang_url, "lang.json")
    lang_properties = json.load(open(json_url))
    return int(lang_properties["maxDocTFLength"])

def get_doc_tf_vector(doc_url):
    json_url = os.path.join(doc_url, "docInfo.json")
    doc_info = json.load(open(json_url))
    freqs = doc_info["freqs"]
    freqs = zip(*[freqs*1])
    return freqs

def get_doc_tf_lang (lang_url):
    max_len = get_docs_max_tf_length(lang_url)
    doc_tf_lang = None
    for f in os.listdir(lang_url):
        d = os.path.join(lang_url, f)
        if os.path.isdir(d):
            doc_tf = get_doc_tf_vector(d)
            extend(doc_tf, max_len)
            if doc_tf_lang == None:
                doc_tf_lang = [doc_tf]
            else:
                doc_tf_lang.append(doc_tf)
    return doc_tf_lang

# ====================================================

def get_docs_max_sizes_length(lang_url):
    json_url = os.path.join(lang_url, "lang.json")
    lang_properties = json.load(open(json_url))
    return int(lang_properties["maxDocSizesLength"])

def get_doc_sizes_vector(doc_url):
    json_url = os.path.join(doc_url, "docInfo.json")
    doc_info = json.load(open(json_url))
    sizes = doc_info["sizes"]
    sizes = zip(*[sizes*1])
    return sizes

def get_doc_sizes_lang (lang_url):
    max_len = get_docs_max_sizes_length(lang_url)
    doc_sizes_lang = None
    for f in os.listdir(lang_url):
        d = os.path.join(lang_url, f)
        if os.path.isdir(d):
            doc_sizes = get_doc_sizes_vector(d)
            extend(doc_sizes, max_len)
            if doc_sizes_lang == None:
                doc_sizes_lang = [doc_sizes]
            else:
                doc_tf_lang.append(doc_tf)
    return doc_tf_lang

# ==========================================================

def get_doc_size(doc_url):
    json_url = os.path.join(doc_url, "docInfo.json")
    doc_info = json.load(open(json_url))
    size = doc_info["size"]
    return size

def get_doc_size_lang (lang_url):
    doc_size_lang = None
    for f in os.listdir(lang_url):
        d = os.path.join(lang_url, f)
        if os.path.isdir(d):
            doc_size = get_doc_size(d)
            if doc_size_lang == None:
                doc_size_lang = [[doc_size]]
            else:
                doc_size_lang.append([doc_size])
    return doc_size_lang

# ====================================================

def get_sents_max_tf_length(lang_url):
    json_url = os.path.join(lang_url, "lang.json")
    lang_properties = json.load(open(json_url))
    return int(lang_properties["maxSentTFLength"])

def get_sents_tf_vector(doc_url):
    json_url = os.path.join(doc_url, "sentTF.json")
    doc_info = json.load(open(json_url))
    freqs = doc_info["freqs"]
    freqs = zip(*[freqs*1])
    return freqs

def get_sents_tf_lang (lang_url):
    max_len = get_sents_max_tf_length(lang_url)
    doc_tf_lang = None
    for f in os.listdir(lang_url):
        d = os.path.join(lang_url, f)
        if os.path.isdir(d):
            doc_tf = get_doc_tf_vector(d)
            extend(doc_tf, max_len)
            if doc_tf_lang == None:
                doc_tf_lang = [doc_tf]
            else:
                doc_tf_lang.append(doc_tf)
    return doc_tf_lang
