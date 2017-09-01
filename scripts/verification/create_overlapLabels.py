import csv
import numpy as np

# Read data from csv files and turn into dictionaries for easy further processing
foxp3Seqs = {}
atacSeqs = {}
labels = np.zeros(63197, np.float)
pkNames = np.zeros(63197, np.int)

with open('../../data/original/Foxp3.ChIPseq.csv') as foxp3:
    reader = csv.reader(foxp3, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in foxp3Seqs:
            foxp3Seqs[row[0]].append((row[1], row[2]))
        else:
            foxp3Seqs[row[0]] = [(row[1], row[2])]

with open('../../data/original/dpz.SplTreg.ATAC.density_GCandquantile.csv') as atac:
    reader = csv.reader(atac, delimiter=',')
    next(reader)    # skip column headers

    i = 1
    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2], 'peak_' + str(i)))
        else:
            atacSeqs[row[0]] = [(row[1], row[2], 'peak_' + str(i))]

        i = i + 1

for chrm, rangeList in foxp3Seqs.items():

    if chrm in atacSeqs.keys():
        for r1 in rangeList:
            s1, e1 = int(r1[0]), int(r1[1])

            for r2 in atacSeqs[chrm]:
                s2, e2 = int(r2[0]), int(r2[1])
                peakLabel = r2[2]
                peakNum = int(peakLabel[5:])

                if max(s1, s2) <= min(e1, e2):
                    overlap = np.abs(min(e1, e2) - max(s1, s2))
                    if overlap == 0:
                        overlap = 1

                    labels[peakNum - 1] = labels[peakNum - 1] + overlap/251

for key, value in atacSeqs.items():
    for tple in value:
        name = tple[2]
        idx = int(name[5:])
        pkNames[idx - 1] = idx

labels = np.delete(labels, np.concatenate(([4296, 23346, 26925, 35323], np.arange(44511, 44515),
                                           np.arange(61378, 61421), np.arange(63177, 63186))))

pkNames = np.delete(pkNames, np.concatenate(([4296, 23346, 26925, 35323], np.arange(44511, 44515),
                                                np.arange(61378, 61421), np.arange(63177, 63186))))

labels.tofile('../../data/overlapLabels', '\n')
pkNames.tofile('../../data/pkNames', '\n')
