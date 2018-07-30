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

PEER_DIR = "/home/kariminf/Data/ATS/Mss15Test/tests/testing2018/"
MODEL_DIR = "/home/kariminf/Data/ATS/Mss15Test/model/"


for lang in os.listdir(PEER_DIR):
    lang_url = os.path.join(PEER_DIR, lang)
    if os.path.isdir(lang_url):
        print lang
        xmlcontent = "<ROUGE-EVAL version=\"1.0\">\n"
        for doc in os.listdir(lang_url):
            xmlcontent += "<EVAL ID=\"" + doc[:-4] + "\">\n"
            xmlcontent += "<PEER-ROOT>\n"
            xmlcontent += PEER_DIR + lang + "\n"
            xmlcontent += "</PEER-ROOT>\n"
            xmlcontent += "<MODEL-ROOT>\n"
            xmlcontent += MODEL_DIR + lang + "\n"
            xmlcontent += "</MODEL-ROOT>\n"
            xmlcontent += "<INPUT-FORMAT TYPE=\"SPL\">\n"
            xmlcontent += "</INPUT-FORMAT>\n"
            xmlcontent += "<PEERS>\n"
            xmlcontent += "<P ID=\"P\">" + doc + "</P>\n"
            xmlcontent += "</PEERS>\n"
            xmlcontent += "<MODELS>\n"
            xmlcontent += "<M ID=\"M"+doc+"\">" + doc + "</M>\n"
            xmlcontent += "</MODELS>\n"
            xmlcontent += "</EVAL>\n"
        xmlcontent += "</ROUGE-EVAL>"

        with open(PEER_DIR + lang + "-2018.xml", "w") as f:
            f.write(xmlcontent)
