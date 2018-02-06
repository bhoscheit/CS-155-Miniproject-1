# Imports data and creates label vector.
# Then, runs a naive bayes classifier.

import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import pdb

# Import data and labels
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
with open('training_data.txt') as f:
    reader = csv.reader(f, delimiter=" ")
    labels = next(reader)


## Data massaging
# option one: 80/20 split.
if False:
    np.random.seed(245)
    np.random.shuffle(original_training_data)
    number_of_data_points = original_training_data.shape[0]
    eighty_percent_cutoff = int(number_of_data_points*0.8)
    test_data = original_training_data[ eighty_percent_cutoff:number_of_data_points, : ]
    training_data = original_training_data[ 0:eighty_percent_cutoff, : ] 
    ## Implement the Naive Bayes classifier.
    #original_x = original_training_data[:,1:]
    #original_y = original_training_data[:,0]
    training_x = training_data[:,1:]
    training_y = training_data[:,0]
    test_x = test_data[:,1:]
    test_y = test_data[:,0]
    alphas = np.linspace(0,10,11)
    for alpha in alphas:
        #gnb = GaussianNB(alpha = alpha)
        gnb = MultinomialNB(alpha = alpha)
        #gnb = BernoulliNB(alpha = alpha)
        gnb.fit( training_x, training_y)
        # Predict labels
        predicted_y = gnb.predict(test_x)
        # Prediction accuracy
        percent_correct = gnb.score(test_x,test_y)
        print(str(percent_correct))
        # Prepare to plot probabilities
        probs = gnb.predict_proba(test_x)
        prob_diffs = probs[:,0] - probs[:,1]
        indices = np.arange(len(prob_diffs))
        indices = np.array(indices)
        prob_diffs_w_id = np.ndarray(shape=(4000,2))
        prob_diffs_w_id[:,0] = indices
        prob_diffs_w_id[:,1] = prob_diffs
        # Plot probabilities
        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        #ax.scatter( prob_diffs_w_id[:,0], prob_diffs_w_id[:,1])
        #plt.show()

# option two: k-fold CV
if True:
    k_folds = 10
    alphas1 = np.logspace(-2,0,20)
    alphas2 = np.logspace(0,2,20)
    alphas = np.append( alphas1, alphas2)
    alphas_results = np.ndarray(shape=( len(alphas), k_folds ) )
    
    kfold = StratifiedKFold(n_splits = k_folds, shuffle=True, random_state=4545)
    kfold.get_n_splits( original_training_data[:,1:], original_training_data[:,0])
    
    current_fold = 0
    for train_index, test_index in kfold.split( original_training_data[:,1:], original_training_data[:,0] ):
        training_data = original_training_data[train_index]
        test_data = original_training_data[test_index]
        ## Implement the Naive Bayes classifier.
        training_x = training_data[:,1:]
        training_y = training_data[:,0]
        test_x = test_data[:,1:]
        test_y = test_data[:,0]
        alpha_results = []
        for alpha in alphas:
            #gnb = GaussianNB()
            #gnb = MultinomialNB(alpha = alpha)
            gnb = BernoulliNB(alpha = alpha)
            gnb.fit( training_x, training_y)
            # Predict labels
            predicted_y = gnb.predict(test_x)
            # Prediction accuracy
            percent_correct = gnb.score(test_x,test_y)
            alpha_results.append(percent_correct)
        alphas_results[:,current_fold] = alpha_results
        current_fold += 1
    # Compute averages for different alphas:
    for alpha_index in range(len(alphas)):
        average_accuracy = np.sum( alphas_results[alpha_index,:] ) / k_folds
        print( "alpha: " + str(alphas[alpha_index]) + \
                " average_accuracy: " + str(average_accuracy) )

    

# Pause the program and inspect the variables
#pdb.set_trace()
