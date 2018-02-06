import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

import csv
import pdb

# Import data and labels
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
with open('training_data.txt') as f:
    reader = csv.reader(f, delimiter=" ")
    labels = next(reader)

# Finish data prep
np.random.seed(245)
np.random.shuffle(original_training_data)
number_of_data_points = original_training_data.shape[0]
eighty_percent_cutoff = int(number_of_data_points*0.8)
test_data = original_training_data[ eighty_percent_cutoff:number_of_data_points, : ] 
training_data = original_training_data[ 0:eighty_percent_cutoff, : ] 
original_x = original_training_data[:,1:]
original_y = original_training_data[:,0]
training_x = training_data[:,1:]
training_y = training_data[:,0]
test_x = test_data[:,1:]
test_y = test_data[:,0]

# Train tf-idf transformer
tfidf = TfidfTransformer()
pdb.set_trace()
