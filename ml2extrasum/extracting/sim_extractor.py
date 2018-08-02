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

import numpy as np
from extractor import Extractor

class SimExtractor(Extractor):

    def __init__(self, sentences, scores, sent_ids, sims):
        super(SimExtractor, self).__init__(sentences, scores)
        self.sent_ids = np.array(sent_ids)[self.order]
        self.sims = sims

    def extract_n_sentences(self, n):
        return self.sorted[:n]

    def extract_n_chars(self, n):
        result = []
        size = 0
        last_id = 0
        for index, sentence in enumerate(self.sorted):

            if index > 0: #the first sentence must be added no matter what
                sent_id = self.sent_ids[index]
                sent_sims = self.sims[sent_id]
                sim_last = sent_sims[last_id]
                last_id = int(sent_id)
                avg_sim = np.mean(sent_sims)
                if sim_last > avg_sim :
                    continue
                size += len(unicode(sentence))
                if size > n:
                    break
            else:
                size += len(unicode(sentence))
            result.append(sentence)
        return result
