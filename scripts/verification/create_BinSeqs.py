import os
import csv
import numpy as np

atacSeqs = {}
seqNum = 1  # track which seq has 'n' or 'N'

with open('../../data/original/dpz.SplTreg.ATAC.density_GCandquantile.csv') as atac:
    reader = csv.reader(atac, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2]))
        else:
            atacSeqs[row[0]] = [(row[1], row[2])]

# Read the sequences from each .fa file using the ranges in the atacSeqs dict, then write those sequences to a txt file
# with one sequence per line.

output = open('../../data/binSeqs.txt', 'w+')

idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

for file in os.listdir('../../data/mm10'):
    if file.endswith('.fa'):

        with open('../../data/mm10/' + file) as fn:
            filename = file[: -3]   # remove .fa extension
            ranges = atacSeqs[filename]

            fn.__next__() # skip first line
            chrm = fn.read()
            chrm = chrm.replace('\n', '')

            for r in ranges:
                onehot = np.zeros([4, 251], np.int)

                start, end = int(r[0]), int(r[1])
                seq = chrm[start:end+1].upper()

                if seq.__contains__('N'):
                    print(seqNum)
                    continue
                else:
                    i = 0
                    for char in seq:
                        onehot[idx[char], i] = 1
                        i = i + 1

                onehot = onehot.reshape([1, 1004])[0]
                onehot = onehot.astype(np.str)
                onehot = " ".join(onehot)

                output.write(onehot + '\n')

                seqNum = seqNum + 1
    else:
        continue

output.close()
