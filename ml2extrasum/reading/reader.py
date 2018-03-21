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
