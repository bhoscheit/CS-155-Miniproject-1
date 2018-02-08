import pdb
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer

# Import training data
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
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

# Train model on tf-idf training data
training_x = training_tf_idf[:,1:]
training_y = training_tf_idf[:,0]
model = LogisticRegression(penalty='l2', C = 0.975)
model.fit( training_x, training_y )

# Use model to predict tf-idf test data
test_result = model.predict( test_tf_idf )

# Alternative where you train on 80% of the training data and test on the remaining 20%
#np.random.shuffle(training_tf_idf)
#training_x = training_tf_idf[0:16000,1:]
#training_y = training_tf_idf[0:16000,0]
#test_x = training_tf_idf[16000:,1:]
#test_y = training_tf_idf[16000:,0]
#model = LogisticRegression(penalty='l2', C = 0.975)
#model.fit( training_x, training_y )
#test_result = model.score( test_x, test_y )
#print( str(test_result) )
