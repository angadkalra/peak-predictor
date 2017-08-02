import csv

# Read data from csv files and turn into dictionaries for easy further processing

blacklistSeqs = {}
foxp3Seqs = {}
dpzSeqs = {}

foxp3SeqsEdited = {}
dpzSeqsEdited = {}

with open('sorted.mm10.blacklistandchrM.csv') as blacklist:
    reader = csv.reader(blacklist, delimiter=',')
    next(reader)    # skip column headers
    for row in reader:
        if row[0] in blacklistSeqs:
            blacklistSeqs[row[0]].append((row[1], row[2]))
        else:
            blacklistSeqs[row[0]] = [(row[1], row[2])]

with open('Foxp3.ChIPseq.csv') as foxp3:
    reader = csv.reader(foxp3, delimiter=',')
    next(reader)    # skip column headers
    for row in reader:
        if row[0] in foxp3Seqs:
            foxp3Seqs[row[0]].append((row[1], row[2]))
        else:
            foxp3Seqs[row[0]] = [(row[1], row[2])]

with open('dpz.SplTreg.ATAC.density_GCandquantile.csv') as dpz:
    reader = csv.reader(dpz, delimiter=',')
    next(reader)    # skip column headers
    for row in reader:
        if row[0] in dpzSeqs:
            dpzSeqs[row[0]].append((row[1], row[2]))
        else:
            dpzSeqs[row[0]] = [(row[1], row[2])]

# Go through blacklist and remove chromosomes from other 2 files that are in/overlap with blacklist

for chrm, rangeList in blacklistSeqs.items():
    foxp3List = foxp3Seqs[chrm]
    dpzList = dpzSeqs[chrm]

    i = 0
    for r1 in rangeList:
        s1, e1 = r1  # change tuple values to ints! not strings

        for r2 in foxp3List:
            s2, e2 = r2
            if (s1 <= e2 and e2 <= e1) or (s1 <= s2 and s2 <= e1):
                foxp3List.remove(r2)

        for r2 in dpzList:
            s2, e2 = r2
            if (s1 <= e2 and e2 <= e1) or (s1 <= s2 and s2 <= e1):
                dpzList.remove(r2)

        foxp3SeqsEdited[chrm] = foxp3List
        dpzSeqsEdited[chrm] = dpzList

print('Done!')
