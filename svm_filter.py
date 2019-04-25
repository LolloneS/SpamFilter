#!/usr/bin/env python3

import numpy as np
import sys
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import sklearn.exceptions
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer, check_scoring
from plot_learning_curve import plot_learning_curve



classification_report_results = {
    'Linear': [],
    'Poly2': [],
    'RBF': []
}


def classification_report_with_accuracy_score(y_true, y_pred, **kwargs):
    # print(classification_report(y_true, y_pred))  # print classification report
    classification_report_results.get(kwargs.get('kwargs')).append(classification_report(y_true, y_pred))
    return accuracy_score(y_true, y_pred)  # return accuracy score


def print_results(r):
    for k, v in r.items():
        print("Minimum value for {}: {}".format(k, v.min()))
        print("Average value for {}: {}".format(k, v.mean()))
        print("Maximum value for {}: {}".format(k, v.max()))
        print('\n')


def tf_idf(x):
    N = x.shape[0]  # number of documents
    idf = np.log10(N / (x != 0).sum(0)) / 100.0
    return x * idf


with open("spambase_clean.data") as f:
    data = np.loadtxt(f, delimiter=',')
    np.random.shuffle(data)
    x = data[:, 0:54]
    y = data[:, 57]
    x = tf_idf(x)
    norms = np.sqrt(((x + 1e-128) ** 2).sum(axis=1, keepdims=True))
    x_normalized = np.where(norms > 0.0, x / norms, 0.)

# SVM classification using linear, poly2 and RBF kernel over TF/IDF representation
classifiers = {
    'Linear': SVC(kernel='linear', gamma='auto'),
    'Poly2': SVC(kernel='poly', gamma='auto', degree=2),
    'RBF': SVC(kernel='rbf', gamma='auto')
}

results, results_normalized = {}, {}

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
title = "Learning Curves (SVM, kernel = {})"

for k, v in classifiers.items():
    # plot = plot_learning_curve(v, title.format(k), x, y, ylim=(0.55, 1.01), cv=cv, n_jobs=-1)
    # plot.show()
    # results[k] = cross_val_score(v, x, y, cv=10, n_jobs=-1,
    #                              scoring=make_scorer(classification_report_with_accuracy_score, kwargs=k))
    results[k] = cross_val_score(v, x, y, cv=10, n_jobs=-1, scoring='accuracy')
    # results_normalized[k] = cross_val_score(v, x_normalized, y, cv=10, n_jobs=-1,
    #                                         scoring=make_scorer(classification_report_with_accuracy_score, kwargs=k))
    results_normalized[k] = cross_val_score(v, x_normalized, y, cv=10, n_jobs=-1, scoring='accuracy')


print("Results using non-normalized input")
print_results(results)

print("Results using angular kernel")
print_results(results_normalized)
