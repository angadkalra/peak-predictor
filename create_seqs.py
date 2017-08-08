import csv
import numpy as np

atacSeqs = {}

with open('data/dpz.SplTreg.ATAC.density_GCandquantile.csv') as atac:
    reader = csv.reader(atac, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2]))
        else:
            atacSeqs[row[0]] = [(row[1], row[2])]

# Read each .fa file into a dictionary with keys as "chrN" and values as a list of strings