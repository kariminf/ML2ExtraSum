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
import reader

DOC_SIM_LIMIT = 200
SENT_SIM_LIMIT = 50

DOC_TF_LIMIT = 200
SENT_TF_LIMIT = 50

DOC_SIZE_LIMIT = 100

class LimitedReader(reader.Reader):
    def __init__(self, url):
        self.url = url

# ====================================================

def list_to_size(list, limit):
    list.sort(reverse=True)
    list = list[:limit]
    extend(list, limit)

# ====================================================

def get_doc_sim_vector(doc_url):
    sims = reader.get_doc_sim_vector(doc_url)
    list_to_size(sims, DOC_SIM_LIMIT)
    return sims

# ====================================================

def get_doc_tf_vector(doc_url):
    freqs = reader.get_doc_tf_vector(doc_url)
    list_to_size(freqs, DOC_TF_LIMIT)
    return freqs

# ====================================================

def get_doc_sizes_vector(doc_url):
    sizes = reader.get_doc_sizes_vector(doc_url)
    list_to_size(sizes, DOC_SIZE_LIMIT)
    return sizes

# ====================================================

def get_sents_tf_vector(doc_url):
    freqs = reader.get_sents_tf_vector(doc_url)
    list_to_size(freqs, SENT_TF_LIMIT)
    return freqs

# ====================================================

def get_sents_sim_vector(doc_url):
    sims = reader.get_sents_sim_vector(doc_url)
    list_to_size(sims, SENT_SIM_LIMIT)
    return sims
