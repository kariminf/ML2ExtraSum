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
import json

sys.path.insert(0, "..")

from ml2extrasum.reading.reader import Reader

r = Reader("/home/kariminf/Data/ATS/Mss15Train/stats/")
r.set_lang("ar")

batch = r.create_doc_batch()

with open('data.json', 'w') as fp:
    json.dump(batch, fp)
