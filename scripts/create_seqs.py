import os
import csv

atacSeqs = {}
seqNum = 1  # track which seq has 'n' or 'N'

with open('../data/original/dpz.SplTreg.ATAC.density_GCandquantile.csv') as atac:
    reader = csv.reader(atac, delimiter=',')
    next(reader)    # skip column headers

    for row in reader:
        if row[0] in atacSeqs:
            atacSeqs[row[0]].append((row[1], row[2]))
        else:
            atacSeqs[row[0]] = [(row[1], row[2])]

# Read the sequences from each .fa file using the ranges in the atacSeqs dict, then write those sequences to a txt file
# with one sequence per line.

output = open('../data/seqsVerify.txt', 'w+')

for file in os.listdir('../data/mm10'):
    if file.endswith('.fa'):

        with open('../data/mm10/' + file) as fn:
            filename = file[: -3]   # remove .fa extension
            ranges = atacSeqs[filename]

            fn.__next__() # skip first line
            chrm = fn.read()
            chrm = chrm.replace('\n', '')

            for r in ranges:

                start, end = int(r[0]), int(r[1])
                seq = chrm[start:end+1].upper()

                if seq.__contains__('N') or seq.__contains__('n'):
                    print(seqNum)
                    continue

                output.write(seq + '\n')
                seqNum = seqNum + 1
    else:
        continue

output.close()