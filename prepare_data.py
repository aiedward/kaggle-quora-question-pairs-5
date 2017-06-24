########################################
## import packages
########################################
import os
import re
import csv
import codecs

import cPickle
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys

from string import maketrans
import keras.preprocessing.text


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]


keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence


reload(sys)
sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = './data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e ?-? ?mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


if not os.path.exists("word_index") or not os.path.exists("data.pkl"):
    ########################################
    ## process texts in datasets
    ########################################
    print('Processing text dataset')

    texts_1 = []
    texts_2 = []
    labels = []
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts_1.append(text_to_wordlist(values[3]))
            texts_2.append(text_to_wordlist(values[4]))
            labels.append(int(values[5]))
    print('Found %s texts in train.csv' % len(texts_1))

    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_texts_1.append(text_to_wordlist(values[1]))
            test_texts_2.append(text_to_wordlist(values[2]))
            test_ids.append(values[0])
    print('Found %s texts in test.csv' % len(test_texts_1))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(labels)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_ids = np.array(test_ids)

    cPickle.dump(word_index, open("word_index", "wb"))
    cPickle.dump(((data_1, data_2, labels), (test_data_1, test_data_2, test_ids)), open("data.pkl", "wb"))
else:
    word_index = cPickle.load(open("word_index", "rb"))
    (data_1, data_2, labels), (test_data_1, test_data_2, test_ids) = cPickle.load(open("data.pkl", "rb"))



print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

np.save("embedding_matrix", embedding_matrix)
