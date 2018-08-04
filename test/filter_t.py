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

#just a code to learn how to sum a vector then divide it on the sum of another
#victor, then multiply by a scalar

import tensorflow as tf

#import os
import sys
sys.path.insert(0, "..")
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml2extrasum.filtering.filter import Filter

import numpy as np

X = [
	[[0.7], [.7], [0.7], [0.2], [0.2]],
	[[0.7], [0.6], [0.4], [0.4], [0.4]],
	[[1.0], [0.4], [0.4], [0.4], [0.3]],
	[[1.0], [0.4], [0.4], [0.4], [0.3]]
]


if __name__ == '__main__':
	x = tf.placeholder(tf.float32, shape=[None,None,1], name="x-input")

	filterer = Filter(x, "filter")

	graph = filterer.get_graph()
	th = filterer.get_threshold()

	sess = tf.Session()

	t, res = sess.run([th, graph], feed_dict={x: X})
	print t, res
