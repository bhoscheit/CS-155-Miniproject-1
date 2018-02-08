import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer

# Import training data
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
train = original_training_data
# Import test data
original_test_data = np.loadtxt("test_data.txt", skiprows=1)
# Produce joint tf-idf-generating dataset
pre_tf_idf = np.append(original_training_data[:,1:], original_test_data, axis=0)

# Compute tf-idf
tfidf_trans = TfidfTransformer()
full_tf_idf = tfidf_trans.fit_transform(pre_tf_idf)
full_tf_idf = full_tf_idf.toarray()

# Reconstitute training data
train_labels = original_training_data[:,0]
train_labels = train_labels.reshape(20000,1)
training_tf_idf = np.append(train_labels, full_tf_idf[0:20000,:], axis=1)

# Reconstitute test data
test_tf_idf = full_tf_idf[20000:, :]

# Implement k-fold cross-validation
k_folds = 10
#sees1 = np.linspace(0.97,0.98,10)
#sees2 = np.linspace(0.98,0.99,10)
sees1 = np.linspace(0.945,0.95,10)
sees2 = np.linspace(0.95,0.955,10)
#sees1 = np.logspace(-2,0,10)
#sees2 = np.logspace(0,2,10)
sees = np.append( sees1, sees2)
sees_results = np.ndarray(shape=( len(sees), k_folds ) ) 

kfold = StratifiedKFold(n_splits = k_folds, shuffle=True, random_state=4525)
kfold.get_n_splits( training_tf_idf[:,1:], training_tf_idf[:,0])

current_fold = 0
for train_index, test_index in kfold.split( training_tf_idf[:,1:], training_tf_idf[:,0] ):
    training_data = training_tf_idf[train_index]
    test_data = training_tf_idf[test_index]
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
