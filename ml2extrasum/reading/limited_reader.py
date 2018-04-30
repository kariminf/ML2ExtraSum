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
from reader import Reader

DOC_SIM_LIMIT = 3
SENT_SIM_LIMIT = 50

DOC_TF_LIMIT = 200
SENT_TF_LIMIT = 50

DOC_SIZE_LIMIT = 100

def list_to_size(vector, limit):
    #vector.sort(reverse=True)
    vector = vector[:limit]
    reader.extend(vector, limit)
    return vector

class LimitedReader(Reader):

    def get_doc_sim_vector(self, doc):
        sims = self.get_vector(doc + "/docInfo.json", "sims")
        return list_to_size(sims, DOC_SIM_LIMIT)

    def get_doc_tf_vector(self, doc):
        freqs = self.get_vector(doc + "/docInfo.json", "freqs")
        return list_to_size(freqs, DOC_TF_LIMIT)

    def get_doc_sizes_vector(self, doc):
        sizes = self.get_vector(doc + "/docInfo.json", "sizes")
        return list_to_size(sizes, DOC_SIZE_LIMIT)

    def get_sents_tf_vector(self, doc):
        freqs = self.get_vector(doc + "/sentTF.json", "freqs")
        return list_to_size(freqs, SENT_SIM_LIMIT)
