from __future__ import unicode_literals

import gensim
import codecs
import argparse
import os
import csv
import re
from nltk.stem import SnowballStemmer
from string import punctuation


stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then',
              'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to',
              'What', 'Which',
              'Is', 'If', 'While', 'This']


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text.split()


class MySentences(object):
    LOG_EVERY = 10000

    def __init__(self, train_filename, test_filename, verbose=True):
        self._filenames = [os.path.expanduser(train_filename), os.path.expanduser(test_filename)]
        self._verbose = verbose

    def __iter__(self):
        cnt = 0
        for i, fname in enumerate(self._filenames):
            if i == 0:
                qidxs = [3,4]
            else:
                qidxs = [1,2]
            with codecs.open(fname, encoding='utf8') as fin:
                reader = csv.reader(fin, delimiter=',')
                header = next(reader)
                for values in reader:
                    yield text_to_wordlist(values[qidxs[0]])
                    yield text_to_wordlist(values[qidxs[1]])

                    cnt += 1
                    if cnt % self.LOG_EVERY == 0 and self._verbose:
                        print("On line #{}".format(cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train word2vec. Input files should be in the format of'
                                                 'lines with sentences.\n\n'
                                                 'Usage: python train_w2v.py '
                                                 '--size=300 --window=2 --workers=16 '
                                                 '--files_prefix=w2v_file_',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train_filename', type=str)
    parser.add_argument('--test_filename', type=str)
    parser.add_argument('--size', type=int, default=300, help='Size of embeddings. Default=300.')
    parser.add_argument('--window', type=int, default=3, help='Context window size. Default=3.')
    parser.add_argument('--sg', type=int, default=0, help='Learning method. 0 means cbow,'
                                                          '1 means skip-gram. Default=cbow.')
    parser.add_argument('--min_count', type=int, default=3, help='Vocabulary word minimum frequency. Default=3.')
    parser.add_argument('--workers', type=int, default=4, help='Num of workers. As greater as faster. Default=4.')
    parser.add_argument('--iter', type=int, default=5, help='Number of epochs. Default=5.')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbosity flag. Default=True.')

    args = parser.parse_args()

    print("Train word2vec with args:")
    print("--size=%d" % args.size)
    print("--window=%d" % args.window)
    print("--sg=%d" % args.sg)
    print("--min_count=%d" % args.min_count)
    print("--workers=%d" % args.workers)
    print("--iter=%d" % args.iter)
    print("--train_filename=%s" % args.train_filename)
    print("--test_filename=%s" % args.test_filename)
    print("--verbose=%d" % args.verbose)

    sentences = MySentences(args.train_filename, args.test_filename, args.verbose)

    print("Start training...")
    word2vec = gensim.models.word2vec.Word2Vec(sentences, size=args.size, window=args.window,
                                               sg=args.sg, min_count=args.min_count,
                                               workers=args.workers, iter=args.iter)
    modelname_base = '_'.join(['size', str(args.size),
                               'window', str(args.window),
                               'sg', str(args.sg),
                               'min_count', str(args.min_count),
                               'iter', str(args.iter)])

    modelname = 'w2v_' + modelname_base + '.model'

    print("Save w2v model to '%s'" % modelname)
    word2vec.save(modelname)
