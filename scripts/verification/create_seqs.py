import os
import csv

atacSeqs = {}

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

# Read the sequences from each .fa file using the ranges in the atacSeqs dict, then write those sequences to a txt file
# with one sequence per line.

output = open('../../data/seqsVerify.txt', 'w+')

for file in os.listdir('../../data/mm10'):
    if file.endswith('.fa'):

        with open('../../data/mm10/' + file) as fn:
            filename = file[: -3]   # remove .fa extension
            ranges = atacSeqs[filename]

            fn.__next__() # skip first line
            chrm = fn.read()
            chrm = chrm.replace('\n', '')

            for r in ranges:

                start, end = int(r[0]), int(r[1])
                seq = chrm[start:end+1].upper()
                peakLabel = r[2]

                if seq.__contains__('N'):
                    print(peakLabel)
                    continue
                else:
                    output.write(seq + '\n')
    else:
        continue

output.close()
