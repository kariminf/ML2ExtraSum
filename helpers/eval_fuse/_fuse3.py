#!/usr/bin/python

# -*- coding: utf-8 -*-

import os
# Import from lpod
from lpod.document import odf_new_document
from lpod.table import odf_create_table, odf_create_row, odf_create_cell


def update_results(csv_path, lang, results):
    csv_file = open(csv_path, 'r')
    lnbr = 0
    lang_res = []
    while 1:
        line = csv_file.readline()
        if line == "":
            break
        lnbr = lnbr + 1
        if lnbr < 3:
            continue
        row = odf_create_row()
        vals = line.split(",")
        lang_res.append(float(vals[1]))
    results[lang] = lang_res

def get_labels(csv_path):
    csv_file = open(csv_path, 'r')
    labels = []
    lnbr = 0
    while 1:
        line = csv_file.readline()
        if line == "":
            break
        lnbr = lnbr + 1
        if lnbr < 3:
            continue
        vals = line.split(",")
        labels.append(vals[0])
    return labels

def create_table(labels, results):

    document = odf_new_document('spreadsheet')
    body = document.get_body()
    table= odf_create_table("results")

    row = odf_create_row()
    rnbr = 1
    for lang in sorted(results):
        cell = odf_create_cell()
        cell.set_value(lang, text = lang)
        row.set_cell(rnbr, cell)
        rnbr += 1
    table.set_row(0, row)

    lnbr = 1

    for i in range(len(labels)):
        row = odf_create_row()
        cell = odf_create_cell()
        cell.set_value(labels[i], text = labels[i])
        row.set_cell(0, cell)
        rnbr = 1
        for lang in sorted(results):
            cell = odf_create_cell()
            val = results[lang][i]
            cell.set_value(val, text = "%.5f" % val, cell_type="float")
            row.set_cell(rnbr, cell)
            rnbr += 1
        table.set_row(lnbr, row)
        lnbr += 1

    body.append(table)
    document.save(target="__res2.ods", pretty=True)

results = {}
for csv_path in sorted(os.listdir('.')):
    if csv_path.endswith("-2018.csv"):
        lang = csv_path[:2]
        print(lang)
        update_results(csv_path, lang, results)

labels = get_labels("en-2018.csv")
create_table(labels, results)
