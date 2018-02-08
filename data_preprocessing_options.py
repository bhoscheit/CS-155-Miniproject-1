# Imports data and creates several modified datasets:
#
# To recreate any of the numbered datasets, just that section and any
# prerequisite sections.
#
# Sections:
# 1) train: just read in entire, unmodified training dataset
# 2) train_verbosity: train + verbosity (raw word sum over each review)
# 3) train_verbosity_perc: train + verbosity percentile over all reviews
#   (e.g., verbosity_perc = .015 means that 1.5% of all reviews have fewer
#   words than this review.)
# 4) train_vocab = train + vocab measurement (% of total available words used)
# 5) train_vocab_perc = train + percentile of vocab
# 6) train_tf_idf = change all word counts to tf-idf values (approximately, word importance
#    within a review, times the rarity of finding that word in any given review)

from scipy.stats import percentileofscore
import numpy as np
import pdb

# 1) Import original data
# prereqs: none
original_training_data = np.loadtxt("training_data.txt", skiprows=1)
train = original_training_data

# 2) Add in verbosity term
# prereq: 1
verbosity = np.sum(train, axis = 1)
verbosity = np.ndarray( shape=(20000,1), buffer=verbosity )
train_verbosity = np.append(train, verbosity, axis=1)

# 3) Add in percentile of verbosity term
# prereq: 1
verbosity = np.sum(train, axis = 1)
verbosity = np.ndarray(shape=(20000,), buffer=verbosity)
verbosity_sorted = np.sort(verbosity)
verbosity_perc = [percentileofscore(verbosity_sorted, a, 'rank') for a in verbosity]
verbosity_perc = np.array( verbosity_perc)
verbosity_perc = np.ndarray( shape=(20000,1), buffer=verbosity_perc)
train_verbosity_perc = np.append(train, verbosity_perc, axis=1)

# 4) Add in vocabulary size (% words from entire corpus present in review)
# prereq: 1
train_binary = np.where(train[:,1:] > 0, 1, 0)
raw_vocab = np.sum(train_binary, axis=1)
vocab = raw_vocab/1000
vocab = np.ndarray( shape=(20000,1), buffer=vocab )
train_vocab = np.append( train, vocab, axis=1 )

# 5) Add in percentile of vocabulary size
# prereq: 1
train_binary = np.where(train[:,1:] > 0, 1, 0)
raw_vocab = np.sum(train_binary, axis=1)
vocab = raw_vocab/1000
vocab_sorted = np.sort(vocab)
vocab_perc = [percentileofscore(vocab_sorted, a, 'rank') for a in vocab]
vocab_perc = np.array(vocab_perc)
vocab = np.ndarray( shape=(20000,1), buffer=vocab )
train_vocab_perc = np.append( train, vocab_perc, axis=1)

# 6) entire dataset converted to tf-idf format
# prereq: 1
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
pdb.set_trace()
train_labels = train_labels.reshape(20000,1)
train_tf_idf = np.append(train_labels, tf_idf, axis=1)
