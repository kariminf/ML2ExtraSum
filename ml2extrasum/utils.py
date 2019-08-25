#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2019 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
#  2019	Abdelkrime Aries <kariminfo0@gmail.com>
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

import json, os

from modeling.stat_net_basic import StatNet as SBasic
from modeling.stat_net_filter import StatNet as SFilter
from modeling.stat_net_norm import StatNet as SNorm
from modeling.stat_net_pure import StatNet as SPure

CONFIG = "config.json"

MODELING = {
    "basic": lambda: SBasic,
    "filter": lambda: SFilter,
    "norm": lambda: SNorm,
    "pure": lambda: SPure
}

def get_config():
    path = os.path.join("./", CONFIG)
    return json.load(open(path))

def get_model(name):
    func = MODELING.get(name, lambda: SBasic)
    print name + " model has been chosen"
    return func()
