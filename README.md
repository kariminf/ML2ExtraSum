# (ML)<sub>2</sub> ExtraSum

[![Project](https://img.shields.io/badge/Project-ML2ExtraSum-0BDA51.svg?style=plastic)](https://github.com/kariminf/ML2ExtraSum)
[![Type](https://img.shields.io/badge/Type-Research-0BDA51.svg?style=plastic)](https://github.com/kariminf/ML2ExtraSum)
[![License](https://img.shields.io/badge/License-Apache_2-0BDA51.svg?style=plastic)](http://www.apache.org/licenses/LICENSE-2.0)


(ML)<sub>2</sub> ExtraSum: Machine Learning based Multi-Lingual Extractive Summarizer

It is a system which uses some surface features extracted from the sentences and their documents to learn ROUGE-1 score of each sentence.
Our idea is to express the problem of text summarization as a problem of regression.
Using some features of a sentence and its document, the system must learn to estimate ROUGE-1 of this sentence based on a reference summary.

![general architecture](assets/archi.png)

As shown in the previous figure, we build some blocks of multi-layer neural networks which we call scorers.
Each scorer receives one type of features; for instance, there is a scorer which scores a sentence based on its term frequencies and those of its document.
All these scorers outputs are fed into another scorer to be combined into one final score which is meant to be an estimation of ROUGE-1 score of the sentence in question.
The system is trained on multiple languages, this is why a block (language clusterer) is reserved to detect the language.
In our case, the language clusterer represent each language as a two dimentional vector which is fed into other scorers so they learn to score based on the language. 

## License
Copyright (C) 2018-2019 Abdelkrime Aries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
