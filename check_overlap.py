import csv
import numpy as np

# Read data from csv files and turn into dictionaries for easy further processing
foxp3Seqs = {}
atacSeqs = {}
labels = np.zeros(63145)

with open('data/Foxp3.ChIPseq.csv') as foxp3:
    reader = csv.reader(foxp3, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in foxp3Seqs:
            foxp3Seqs[row[0]].append((row[1], row[2]))
        else:
            foxp3Seqs[row[0]] = [(row[1], row[2])]

with open('data/dpz.SplTreg.ATAC.density_GCandquantile.csv') as dpz:
    reader = csv.reader(dpz, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2]))
        else:
            atacSeqs[row[0]] = [(row[1], row[2])]

i = 0
for chrm, rangeList in foxp3Seqs.items():

    for r1 in rangeList:
        s1, e1 = int(r1[0]), int(r1[1])

        if chrm in atacSeqs.keys():
            j = i

            for r2 in atacSeqs[chrm]:
                s2, e2 = int(r2[0]), int(r2[1])

                if max(s1, s2) <= min(e1, e2):
                    labels[j] = 1

                j = j + 1

    i = i + len(atacSeqs[chrm])

labels = labels.astype(np.int)
np.savetxt('data/labels', labels)

print('Done')