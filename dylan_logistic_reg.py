import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer

# Import data
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
train = original_training_data

# Compute tf-idf
if False:
    ## compute tf
    train_row_sums = np.sum(train[:,1:], axis=1) # don't want to get the labels in here
    tf = train[:,1:] / train_row_sums[:, np.newaxis]
    ## compute idf
    train_binary = np.where(train[:,1:] > 0, 1, 0) # or here
    train_binary_column_sums = np.sum(train_binary, axis=0)
    idf = train_binary_column_sums / 20000
    # compute tf_idf
    tf_idf = tf*idf
    # compute train_tf_idf
    train_labels = train[:,0]
    train_labels = train_labels.reshape(20000,1)
    train_tf_idf = np.append(train_labels, tf_idf, axis=1)
    original_training_data = train_tf_idf
if True:
    tfidf_trans = TfidfTransformer()
    tf_idf = tfidf_trans.fit_transform(train[:,1:])
    tf_idf = tf_idf.toarray()
    train_labels = train[:,0]
    train_labels = train_labels.reshape(20000,1)
    train_tf_idf = np.append(train_labels, tf_idf, axis=1)
    original_training_data = train_tf_idf


# Implement k-fold cross-validation
k_folds = 10
sees1 = np.linspace(0.97,0.98,10)
sees2 = np.linspace(0.98,0.99,10)
#sees1 = np.logspace(-2,0,20)
#sees2 = np.logspace(0,2,20)
sees = np.append( sees1, sees2)
sees_results = np.ndarray(shape=( len(sees), k_folds ) ) 

kfold = StratifiedKFold(n_splits = k_folds, shuffle=True, random_state=4525)
kfold.get_n_splits( original_training_data[:,1:], original_training_data[:,0])

current_fold = 0
for train_index, test_index in kfold.split( original_training_data[:,1:], original_training_data[:,0] ):
    training_data = original_training_data[train_index]
    test_data = original_training_data[test_index]
    training_x = training_data[:,1:]
    training_y = training_data[:,0]
    test_x = test_data[:,1:]
    test_y = test_data[:,0]
    see_results = []
    for see in sees:
        model = LogisticRegression(penalty='l2', C = see)
        model.fit( training_x, training_y )
        see_result =  model.score( test_x, test_y)
        see_results.append(see_result)
    see_results = np.array(see_results)
    sees_results[:,current_fold] = see_results
    current_fold += 1
for see_index in range(len(sees)):
    average_accuracy = np.sum( sees_results[see_index,:] ) / k_folds
    print( "C: " + str(sees[see_index]) + \
            " average_accuracy: " + str(average_accuracy) )

#average_accuracy = np.average(fold_accuracy)
#print( "average accuracy: " + str(average_accuracy) )
