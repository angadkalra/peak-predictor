import csv
import numpy as np

# Read data from csv files and turn into dictionaries for easy further processing
foxp3Seqs = {}
atacSeqs = {}
labels = np.zeros(63197, np.float)

with open('../data/original/Foxp3.ChIPseq.csv') as foxp3:
    reader = csv.reader(foxp3, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in foxp3Seqs:
            foxp3Seqs[row[0]].append((row[1], row[2]))
        else:
            foxp3Seqs[row[0]] = [(row[1], row[2])]

with open('../data/original/dpz.SplTreg.ATAC.density_GCandquantile.csv') as atac:
    reader = csv.reader(atac, delimiter=',')
    next(reader)    # skip column headers

    i = 1
    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2], 'peak_' + str(i)))
        else:
            atacSeqs[row[0]] = [(row[1], row[2], 'peak_' + str(i))]

        i = i + 1


posPks = open('../data/posPksVerify', 'w+')

for chrm, rangeList in foxp3Seqs.items():

    if chrm in atacSeqs.keys():
        for r1 in rangeList:
            s1, e1 = int(r1[0]), int(r1[1])

            for r2 in atacSeqs[chrm]:
                s2, e2 = int(r2[0]), int(r2[1])
                peakLabel = r2[2]
                peakNum = int(peakLabel[5:])

                if max(s1, s2) <= min(e1, e2) and not (labels[peakNum - 1] == 1):
                    labels[peakNum - 1] = 1
                    if not chrm.__contains__('_'):
                        posPks.write(r2[2] + '\n')

posPks.close()

labels = np.delete(labels, np.concatenate(([4296, 23345, 26923, 35320], np.arange(44512, 44516),
                                           np.arange(61379, 61422), np.arange(63178, 63187))))

labels.tofile('../data/labelsVerify', '\n')
