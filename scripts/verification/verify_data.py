import numpy as np
import scipy.io as sio

# Extract posPks from matlab file

org_data = sio.loadmat('../../data/original/peaksATGC.mat')
pkNames = org_data['pkName']
org_labels = org_data['labels']

labelsVerify = np.loadtxt('../../data/labelsVerify')

indices = org_labels.astype(np.bool)

posPkNames = pkNames[indices]

with open('../data/posPks', 'w+') as file:
    for pk in posPkNames:
        pkName = pk[0]
        file.write(pkName + '\n')
