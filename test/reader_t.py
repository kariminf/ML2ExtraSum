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

#just a code to learn and test (a ==> b)

import sys
import numpy
import pytest

sys.path.insert(0, "..")

from ml2extrasum.reading.reader import Reader

r = Reader("/home/kariminf/Data/ATS/Mss15Train/stats/")
r.set_lang("ar")

def test_get_doc_sim_lang():
    assert r.get_docs_max_sim_length() == 29869
    m = r.get_doc_sim_lang()
    assert numpy.shape(m) == (30, 29869, 1)

def test_get_doc_tf_lang():
    assert r.get_docs_max_tf_length() == 1377
    m = r.get_doc_tf_lang()
    assert numpy.shape(m) == (30, 1377, 1)

def test_get_doc_sizes_lang():
    assert r.get_docs_max_sizes_length() == 321
    m = r.get_doc_sizes_lang()
    assert numpy.shape(m) == (30, 321, 1)

def test_get_sents_tf_lang():
    assert r.get_sents_max_tf_length() == 202
    m = r.get_sents_tf_lang()
    assert numpy.shape(m) == (30, 202, 1)
