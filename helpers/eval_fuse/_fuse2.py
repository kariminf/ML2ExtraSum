#!/usr/bin/python

# -*- coding: utf-8 -*-

import os
# Import from lpod
from lpod.document import odf_new_document
from lpod.table import odf_create_table, odf_create_row, odf_create_cell


def createTable(csv_path, table):
	csv_file = open(csv_path, 'r')
	lnbr = 0
	while 1:
		line = csv_file.readline()
		if line == "":
			break
		#lnbr = lnbr + 1
		row = odf_create_row()
		vals = line.split(",")
		rnbr = 0
		for val in vals:
			cell = odf_create_cell()
			if lnbr < 2 or rnbr < 1:
				#table.set_value((rnbr, lnbr), val)
				cell.set_value(val, text = val)
			else:
				#table.set_value((rnbr, lnbr), float(val))
				cell.set_value(float(val), text = "%.5f" % float(val), cell_type="float")
			row.set_cell(rnbr, cell)
			rnbr = rnbr + 1
		table.set_row(lnbr, row)
		lnbr = lnbr + 1


document = odf_new_document('spreadsheet')
body = document.get_body()
 
for csv_path in sorted(os.listdir('.')):
	if csv_path.endswith("-2018.csv"):
		lang = csv_path[:2]
		print(lang)
		table= odf_create_table(lang)
		createTable(csv_path, table)
		body.append(table)
		
document.save(target="__res.ods", pretty=True)
