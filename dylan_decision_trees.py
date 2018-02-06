# Imports data and creates label vector.

import numpy as np
import csv
import sklearn.tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

import pdb

# Import data and labels
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
with open('training_data.txt') as f:
    reader = csv.reader(f, delimiter=" ")
    labels = next(reader)

## Option 1) Split data into test and training subsets
if False:
    np.random.seed(34)
    np.random.shuffle(original_training_data)
    number_of_data_points = original_training_data.shape[0]
    eighty_percent_cutoff = int(number_of_data_points*0.8)
    test_data = original_training_data[ eighty_percent_cutoff:number_of_data_points, : ]
    training_data = original_training_data[ 0:eighty_percent_cutoff, : ]
    # Implement a decision tree named tiny_tree
    tiny_tree = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes = 4)
    tiny_tree.fit(training_data[:,1:],training_data[:,0])
    test_predictions = tiny_tree.predict(test_data[:,1:])
    number_of_rows = test_data.shape[0]
    total_correct = 0
    for row in range(number_of_rows):
        if test_data[row,0]==test_predictions[row]:
            total_correct += 1
    percent_correct = total_correct/number_of_rows
    print(str(percent_correct))
    # Adaboosting based on the tiny_tree
    adabooster = AdaBoostClassifier(base_estimator=tiny_tree,
        n_estimators = 200,
        learning_rate = 0.5)
    adabooster.fit(training_data[:,1:],training_data[:,0])
    adaboost_predictions = adabooster.predict(test_data[:,1:])
    number_of_rows = test_data.shape[0]
    total_correct = 0
    for row in range(number_of_rows):
        if test_data[row,0]==adaboost_predictions[row]:
            total_correct += 1
    percent_correct = total_correct/number_of_rows
    print(str(percent_correct))

## Option 2) K-fold cross validation of bagged decision trees
if True:
    kfold = StratifiedKFold(n_splits = 5)
    kfold.get_n_splits( original_training_data[:,1:], original_training_data[:,0])
    for train_index, test_index in kfold.split( original_training_data[:,1:], original_training_data[:,0] ):
        training_data = original_training_data[train_index]
        test_data = original_training_data[test_index]
        # Implement a decision tree named tiny_tree
        tiny_tree = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes = 4)
        tiny_tree.fit(training_data[:,1:],training_data[:,0])
        test_predictions = tiny_tree.predict(test_data[:,1:])
        number_of_rows = test_data.shape[0]
        total_correct = 0
        for row in range(number_of_rows):
            if test_data[row,0]==test_predictions[row]:
                total_correct += 1
        percent_correct = total_correct/number_of_rows
        print(str(percent_correct))
        # Adaboosting based on the tiny_tree
        adabooster = AdaBoostClassifier(base_estimator=tiny_tree,
            n_estimators = 200,
            learning_rate = 0.5)
        adabooster.fit(training_data[:,1:],training_data[:,0])
        adaboost_predictions = adabooster.predict(test_data[:,1:])
        number_of_rows = test_data.shape[0]
        total_correct = 0
        for row in range(number_of_rows):
            if test_data[row,0]==adaboost_predictions[row]:
                total_correct += 1
        percent_correct = total_correct/number_of_rows
        print(str(percent_correct))



# Pause the program and inspect the variables
pdb.set_trace()
