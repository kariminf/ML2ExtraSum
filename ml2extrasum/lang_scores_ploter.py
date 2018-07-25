#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from numpy.random import rand
import json, os

def get_lang_scores(lang_data):
    X = []
    Y = []
    for doc in lang_data:
        lang_score = lang_data[doc]["lang"]
        X.append(lang_score[0])
        Y.append(lang_score[1])
    return X, Y

colors = [
"red",
"green",
"blue",
"yellow",
"magenta",
"cyan",
"orange",
"hotpink",
"sienna",
"olive",
"gold"
]

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

path = os.path.join("./outputs/", "test.json")

cont = json.load(open(path))

i = 0

fig, ax = plt.subplots()

for lang in cont:
    X, Y = get_lang_scores(cont[lang])
    ax.scatter(X, Y, label=lang, marker=markers[i], facecolors="none", edgecolors=colors[i])
    i += 1

"""
fig, ax = plt.subplots()
for color in ['red', 'green', 'blue']:
    n = 750
    x, y = rand(2, n)
    scale = 200.0 * rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')
"""
#plt.xlim(0.24, 0.26)
#plt.ylim(0.275, 0.290)
ax.legend()
ax.grid(True)

plt.show()
