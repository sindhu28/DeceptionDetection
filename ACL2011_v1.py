#!/usr/bin/env python

from __future__ import print_function

import argparse
import codecs
import logging
import os
import random
import re
import sys
from unixnlp.sys_utils import *


from collections import Counter
from os import path

from nltk import word_tokenize
from nltk.util import ngrams
from numpy import argsort
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

from unixnlp.cat_ASCII import cat_ASCII
from unixnlp.similarity import l2_norm
from unixnlp.split_folds import split_folds
from unixnlp.wordmap import WordMap

DEFAULT_LIWC_HOME = path.join(path.dirname(__file__), "lib", "LIWC")

def LIWC(file, LIWC_home=DEFAULT_LIWC_HOME):
    LIWC_sh = path.join(LIWC_home, "LIWC.sh")
    cmd = " ".join([LIWC_sh, file])
    p = Command(cmd, shell=True, universal_newlines=True)
    (retcode, stdout, stderr) = p.run()

    # extract features, unknown from stdout.
    features = {}
    unknown = {}
    location = 0
    for line in stdout.splitlines():
        if line.startswith("Total number of words:"):
            wc = int(line[line.find(":")+1:])
            if wc == 0:
                wc = 0.0000001
        elif line.startswith("Categories:"):
            location = 1
        elif line.startswith("Unknown words:"):
            location = 2
        elif location == 1:
            (feature, count, percent) = line.strip().replace(":", "").split()
            features[feature] = (int(count), percent)
        elif location == 2:
            (feature, count) = line.strip().replace(":", "").split()
            unknown[feature] = int(count)

    # get tokens to set remaining features. 
    lines = []
    with open(file) as h:
        for line in h:
            lines.append(line.split())
    features["wc"] = (wc, "100%")
    features["wps"] = (wc / len(lines), "{:.1%}".format(1 / len(lines)))
    fdict = wc - sum(unknown.values())
    features["dict"] = (fdict, "{:.1%}".format(fdict / wc))
    fsixltr = sum([
        sum([
            1 for token in line
            if len(token) > 6
        ]) for line in lines])
    features["sixltr"] = (fsixltr, "{:.1%}".format(fsixltr / wc))
    return (features, unknown)


def main():
    parser = argparse.ArgumentParser(description="Replicates ACL 2011 results.")
    parser.add_argument("--debug", action="store_true",
                        help="debug output")
    parser.add_argument("--featuremap", metavar="FILE", default="featuremap",
                        help="location of feature map")
    parser.add_argument("--nfolds", metavar="N", type=int, default=5,
                        help="number of folds to use for cross-validation")
    parser.add_argument("--output", metavar="FILE", default="CV_output",
                        help="file prefix for saving CV output")
    parser.add_argument("--top", metavar="N", type=int, default=0,
                        help="output top N features for each class")
    parser.add_argument("data",
                        help="directory containing review data")
    args = parser.parse_args()

    # set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # load featuremap
    featuremap = WordMap(args.featuremap)

    # load documents, ignoring the folds
    if not path.isdir(args.data):
        raise IOError("Data directory not found: {0}".format(args.data))
    docs = {}
    for (dirpath, dirnames, filenames) in os.walk(args.data):
        for filename in filenames:
            if filename.endswith(".txt"):
                # example filename: t_hilton_1.txt, meaning truthful review of the Hilton hotel
                (label, hotel, i) = filename.split("_")

                # get review text
                with codecs.open(path.join(dirpath, filename), 'r', 'utf8') as h:
                    text = h.read().strip()
                
                # save review
                docs.setdefault(hotel, []).append({
                    '_id': path.join(dirpath, filename),
                    'class': {'t': -1, 'd': 1}.get(label),
                    'hotel': hotel,
                    'text': text
                })

    # split docs into folds (stratified by hotel)
    folds = []
    hotels = docs.keys()
    random.shuffle(hotels)
    for (i, hotels_i) in enumerate(split_folds(hotels, args.nfolds)):
        fold = DocumentCollection()
        for hotel in hotels_i:
            fold += DocumentCollection(docs[hotel])
        folds.append(fold)

    # split data into folds (no stratification)
    #folds = map(DocumentCollection, split_folds(docs, args.nfolds))

    # perform cross validation
    CV_ids, CV_hotels, CV_labels, CV_preds, CV_binary_preds = [], [], [], [], [] # CV output
    for (i, (train, test)) in enumerate(DocumentCollection.get_train_test(folds)):
        # get training features and labels
        logging.getLogger("fold_{0}".format(i)).debug("getting training features and labels")
        train_features = get_features(train, featuremap, is_train=True)
        (train_labels, _) = train.flatten()

        # choose parameters using nested cross-validation on train folds and grid search
        logging.getLogger("fold_{0}".format(i)).debug("tuning parameters")
        grid = GridSearchCV(LinearSVC(loss='l1'),
                            {'C': [0.001, 0.002, 0.004, 0.006, 0.008,
                                   0.01, 0.02, 0.04, 0.06, 0.08,
                                   0.1, 0.2, 0.4, 0.6, 0.8,
                                   1, 2, 4, 6, 8,
                                   10, 20, 40, 60, 80,
                                   100, 200, 400, 600, 800, 1000]}, cv=args.nfolds)
        grid.fit(train_features, train_labels)
        classifier = grid.best_estimator_

        # train classifier using best parameters
        logging.getLogger("fold_{0}".format(i)).debug("training w/ params ({0}) and labels ({1})".format(classifier, Counter(train_labels)))
        classifier.fit(train_features, train_labels)

        # get top N features
        if args.top != 0:
            logging.getLogger("fold_{0}".format(i)).debug("getting top {0} features per class".format(args.top))
            for (label, topN) in get_top_features(classifier, args.top).items():
                logging.getLogger("fold_{0}".format(i)).info("class {0}:".format(label))
                for id in topN:
                    logging.getLogger("fold_{0}".format(i)).info(featuremap[id + 1])

        # get test features and labels
        logging.getLogger("fold_{0}".format(i)).debug("getting test features and labels")
        test_features = get_features(test, featuremap, is_train=False)
        (test_labels, _) = test.flatten()

        # test
        logging.getLogger("fold_{0}".format(i)).debug("testing (labels: {0})".format(Counter(test_labels)))
        test_preds = list(map(lambda y: y[0], classifier.decision_function(test_features)))

        # save CV output
        CV_ids.extend(test['_id'])
        CV_hotels.extend(test['hotel'])
        CV_labels.extend(test_labels)
        CV_preds.extend(test_preds)
        CV_binary_preds.extend(map(lambda y: 1 if y > 0 else -1, test_preds))

    # output various CV reports
    logging.getLogger("CV").info(metrics.classification_report(CV_labels, CV_binary_preds))

    logging.getLogger("CV").info("Per-class (P)recision, (R)ecall, (F)-score")
    logging.getLogger("CV").info("\t-1\t1")
    (precision, recall, fscore, support) = metrics.precision_recall_fscore_support(CV_labels, CV_binary_preds)
    for (metric, results) in [("P", precision), ("R", recall), ("F", fscore)]:
        logging.getLogger("CV").info("{0}\t{1}\t{2}".format(metric, results[0], results[1]))

    logging.getLogger("CV").info("confusion matrix:")
    logging.getLogger("CV").info("real\\pred\t-1\t1")
    (c0, c1) = metrics.confusion_matrix(CV_labels, CV_binary_preds)
    for (label, results) in [("-1", c0), ("1", c1)]:
        logging.getLogger("CV").info("{0}\t{1}\t{2}".format(label, results[0], results[1]))

    # save CV predictions
    if args.output:
        logging.getLogger("CV").debug("saving CV output to {0}".format(args.output))
        with codecs.open(args.output, 'w', 'utf8') as h:
            for (id, hotel, label, pred) in zip(CV_ids, CV_hotels, CV_labels, CV_preds):
                print("{0}\t{1}\t{2}\t{3:f}".format(id, hotel, label, pred), file=h)

    # safe featuremap for future use/reference
    featuremap.write()

def get_features(dc, featuremap, is_train=True):
    row, col, data = [], [], [] # for scikit-learn feature representation

    for (d, (label, text)) in enumerate(dc):
        # start with n-grams
        features = get_ngrams(text, [1,2])

        #add LIWC features
        fopen = open("liwc.txt","w")
        fopen.writelines(text)
        fopen.close()
        (liwc_features, _) = LIWC("liwc.txt")
        new_features = get_liwcFeatures(liwc_features)
        new_features = unit_normalize(new_features)
        
        # unit normalization
        features = unit_normalize(features)
        features.update(new_features)

        # convert features to scikit-learn format
        for (feature, value) in features.items():
            # map features to identifiers
            id = None
            if feature in featuremap or is_train: # ignore features not seen in training
                id = featuremap[feature] - 1

            # add feature value to scikit-learn representation
            if id is not None:
                row.append(d)
                col.append(id)
                data.append(value)

    return coo_matrix( (data, (row, col) ), shape=[len(dc), len(featuremap)] ).tocsr()

def get_liwcFeatures(liwc_features):
    features = {}
    for feature in liwc_features:
        features.update({"LIWC_"+feature: liwc_features[feature][0]})
    return features
    

def get_ngrams(text, N):
    text = cat_ASCII(text)
    text = re.sub("\s+", " ", text)
    text = text.lower()

    features = Counter()
    tokens = word_tokenize(text)
    for n in N:
        for ngram in ngrams(tokens, n):
            feature = "{0}GRAMS_{1}".format(n, "__".join(ngram))
            features.update({feature: 1})
    return features

def unit_normalize(features):
    norm = l2_norm(features)
    return dict([(k, v / norm) for (k, v) in features.items()])

# extract the top N weighted features from the model learned by classifier
def get_top_features(classifier, N):
    # get top features for each class
    sorted_coefs = argsort(classifier.coef_[0])
    return {
        -1: sorted_coefs[:N].tolist(),
        1: reversed(sorted_coefs[-N:].tolist())
    }

class DocumentCollection:
    def __init__(self, collection=[], label_key="class", text_key="text"):
        self.collection = list(collection)
        self.label_key = label_key
        self.text_key = text_key

    def __add__(self, other):
        if self.label_key != other.label_key or self.text_key != other.text_key:
            raise Exception("Mismatched keys!")
        return DocumentCollection(self.collection + other.collection, self.label_key, self.text_key)

    def __iadd__(self, other):
        if self.label_key != other.label_key or self.text_key != other.text_key:
            raise Exception("Mismatched keys!")
        self.collection.extend(other.collection)
        return self

    def __getitem__(self, key):
        return [doc[key] if key else None
                for doc in self.collection]

    def __len__(self):
        return len(self.collection)

    def __iter__(self):
        for doc in self.collection:
            yield (self._label(doc), self._text(doc))
        raise StopIteration

    def _label(self, doc):
        return doc[self.label_key]

    def _text(self, doc):
        return doc[self.text_key]

    def flatten(self):
        return zip(*iter(self))

    @staticmethod
    def get_train_test(folds):
        for (test_i, test) in enumerate(folds):
            train_folds = [folds[i] for i in range(len(folds)) if i != test_i]
            train = reduce(lambda a, b: a + b, train_folds, DocumentCollection())
            yield (train, test)

if __name__ == '__main__':
    main()
