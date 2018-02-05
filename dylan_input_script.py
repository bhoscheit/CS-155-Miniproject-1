# Imports data and creates label vector.

import numpy as np
import csv

#import pdb

training_data = np.loadtxt("training_data.txt", skiprows=1)
with open('training_data.txt') as f:
    reader = csv.reader(f, delimiter=" ")
    labels = next(reader)
#pdb.set_trace()
