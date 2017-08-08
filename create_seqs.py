import os
import csv
import numpy as np

atacSeqs = {}
seqs = []

with open('data/dpz.SplTreg.ATAC.density_GCandquantile.csv') as atac:
    reader = csv.reader(atac, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2]))
        else:
            atacSeqs[row[0]] = [(row[1], row[2])]

# Read the sequences from each .fa file using the ranges in the atacSeqs dict, then write those sequences to a txt file
# with one sequence per line.

for file in os.listdir('data/mm10'):
    if file.endswith('.fa'):
        # match filename with atacSeqs key, then go through ranges and read file and extract sequence for a given range.
        # Then append that seq to seqs[].
    else:
        continue

np.savetxt('data/seqs', seqs)