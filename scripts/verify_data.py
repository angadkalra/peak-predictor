import numpy as np
import scipy.io as sio

# Extract posPks from matlab file

peaksDict = sio.loadmat('../data/original/peaksATGC.mat')
pkNames = peaksDict['pkName']
labels = peaksDict['labels']

labels = labels.astype(np.bool)

posPkNames = pkNames[labels]

with open('../data/posPks', 'w+') as file:
    for pk in posPkNames:
        pkName = pk[0]
        file.write(pkName + '\n')


# Extract posPks from labelsVerify

labels = np.loadtxt('../data/labelsVerify')
posPksVerify = np.argwhere(labels)

with open('../data/posPksVerify', 'w+') as file:
    for pk in posPksVerify:
        pklabel = 'peak_' + str(pk[0] + 1)
        file.write(pklabel + '\n')


