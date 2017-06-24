import cPickle
import os
import re
import csv
import codecs

from datetime import datetime
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Convolution1D, GlobalMaxPooling1D, Merge, \
    merge
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = './'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# num_lstm = 300
# num_dense = 600
# rate_drop_lstm = 0.3
# rate_drop_dense = 0.3

num_lstm = int(sys.argv[1])
num_dense = int(sys.argv[2])
rate_drop_lstm = float(sys.argv[3])
rate_drop_dense = float(sys.argv[4])

re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

# dtime = datetime.now().strftime("%m-%d %H:%M:%S")
STAMP = 'lstm_300_600_0.30_0.30_06-05 20:42:35'

# Load data
(data_1, data_2, labels), (test_data_1, test_data_2, test_ids) = cPickle.load(open("data.pkl", "rb"))

embedding_matrix = np.load("embedding_matrix.npy")
nb_words = embedding_matrix.shape[0]

########################################
## sample train/validation data
########################################
# np.random.seed(42)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

# filters = (num_lstm, num_lstm, num_lstm, num_lstm)
# kernel_sizes = (2, 3, 4, 5)
# cnns1 = []
# cnns2 = []
# for num_filters, kernel_size in zip(filters, kernel_sizes):
#     cnns1.append(
#         Convolution1D(filters=num_filters,
#                       kernel_size=kernel_size)(embedded_sequences_1)
#     )
#     cnns2.append(
#         Convolution1D(filters=num_filters,
#                       kernel_size=kernel_size)(embedded_sequences_2)
#     )
# if len(cnns1) > 1:
#     x1 = concatenate(cnns1, axis=1)
#     y1 = concatenate(cnns2, axis=1)
# else:
#     x1 = cnns1[0]
#     y1 = cnns1[0]
#
# x1 = Activation("tanh")(x1)
# y1 = Activation("tanh")(y1)
#
# x1 = GlobalMaxPooling1D()(x1)
# y1 = GlobalMaxPooling1D()(y1)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)

merged = Dense(num_dense)(merged)
merged = Activation("relu") (merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam')
#model.summary()
print(STAMP)

# early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = "models_lstm/" + STAMP + '.h5'
# model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
#
# hist = model.fit([data_1_train, data_2_train], labels_train, \
#         validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
#         epochs=50, batch_size=2086, shuffle=True, \
#         class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
# bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([data_1, data_2], batch_size=1024, verbose=1)

submission = pd.DataFrame({'lstm':preds.ravel()})
submission.to_csv('data/'+STAMP+'.csv', index=False)
