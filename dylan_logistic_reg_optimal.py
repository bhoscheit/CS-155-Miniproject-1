import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer

# Import data
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
train = original_training_data

# Compute tf-idf
tfidf_trans = TfidfTransformer()
tf_idf = tfidf_trans.fit_transform(train[:,1:])
tf_idf = tf_idf.toarray()
train_labels = train[:,0]
train_labels = train_labels.reshape(20000,1)
train_tf_idf = np.append(train_labels, tf_idf, axis=1)
original_training_data = train_tf_idf

# Train model on tf-idf data
training_x = original_training_data[:,1:]
training_y = original_training_data[:,0]
model = LogisticRegression(penalty='l2', C = 0.975)
model.fit( training_x, training_y )
#test_result =  model.score( test_x, test_y)
