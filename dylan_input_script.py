# Imports data and creates label vector.

import numpy as np
import csv

import pdb

# Import data and labels
training_data = np.loadtxt("training_data.txt", skiprows=1)
with open('training_data.txt') as f:
    reader = csv.reader(f, delimiter=" ")
    labels = next(reader)

# Split data into test an training subsets
np.random.shuffle(training_data)
number_of_data_points = training_data.shape[0]
eighty_percent_cutoff = int(number_of_data_points*0.8)
test_data = training_data[ eighty_percent_cutoff:number_of_data_points, : ]
training_data = training_data[ 0:eighty_percent_cutoff, : ]

# Pause the program and inspect the variables
pdb.set_trace()
