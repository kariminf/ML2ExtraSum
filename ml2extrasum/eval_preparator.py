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
import utils

config = utils.get_config()

SCORE = ["sent", "tf", "sim", "pos", "size", "rouge1", "sums"]
EXTR = ["plain", "sim"]


for lang in os.listdir(config["SUM_DIR"]):
    lang_url = os.path.join(config["SUM_DIR"], lang)
    if os.path.isdir(lang_url) and not lang.startswith("_"):
        print lang
        xmlcontent = "<ROUGE-EVAL version=\"1.0\">\n"
        for doc in os.listdir(lang_url):
            doc_url = os.path.join(lang_url, doc)
            if os.path.isdir(doc_url):
                xmlcontent += "<EVAL ID=\"" + doc + "\">\n"
                xmlcontent += "<PEER-ROOT>\n"
                xmlcontent += config["SUM_DIR"] + lang + "/" + doc + "\n"
                xmlcontent += "</PEER-ROOT>\n"
                xmlcontent += "<MODEL-ROOT>\n"
                xmlcontent += config["REF_DIR"] + lang + "\n"
                xmlcontent += "</MODEL-ROOT>\n"
                xmlcontent += "<INPUT-FORMAT TYPE=\"SPL\">\n"
                xmlcontent += "</INPUT-FORMAT>\n"
                xmlcontent += "<PEERS>\n"
                for score in SCORE:
                    for extr in EXTR:
                        pname = score + "_" + extr
                        xmlcontent += "<P ID=\"" + pname + "\">" + pname + ".txt</P>\n"
                xmlcontent += "</PEERS>\n"
                xmlcontent += "<MODELS>\n"
                xmlcontent += "<M ID=\"M"+doc+"\">" + doc + ".txt</M>\n"
                xmlcontent += "</MODELS>\n"
                xmlcontent += "</EVAL>\n"
        xmlcontent += "</ROUGE-EVAL>"

        with open(config["SUM_DIR"] + lang + config["EXT"] + ".xml", "w") as f:
            f.write(xmlcontent)
