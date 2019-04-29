#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels





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

# Split the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
y_test = np.array(y_test)
y_train = np.array(y_train)
y_test = y_test.astype(int)
y_train = y_train.astype(int)
x_train_normalized, x_test_normalized, y_train_normalized, y_test_normalized = train_test_split(x_normalized, y, random_state=0)
y_test_normalized = np.array(y_test_normalized)
y_train_normalized = np.array(y_train_normalized)
y_test_normalized = y_test_normalized.astype(int)
y_train_normalized = y_train_normalized.astype(int)

y_pred = {
}
y_pred_normalized = {
}

for k, v in classifiers.items():
    y_pred[k] = classifiers.get(k).fit(x_train, y_train).predict(x_test)
    y_pred_normalized[k] = classifiers.get(k).fit(x_train, y_train).predict(x_test)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

print("Confusion matrix of non-normalized vectors")
for k, v in classifiers.items():
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred[k], classes=y_test,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred[k], classes=y_test, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()

print("Confusion matrix of normalized vectors")
for k, v in classifiers.items():
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test_normalized, y_pred_normalized[k], classes=y_test_normalized,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test_normalized, y_pred_normalized[k], classes=y_test_normalized, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

# results, results_normalized = {}, {}
#
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# title = "Learning Curves (SVM, kernel = {})"
#
# for k, v in classifiers.items():
#     # plot = plot_learning_curve(v, title.format(k), x, y, ylim=(0.55, 1.01), cv=cv, n_jobs=-1)
#     # plot.show()
#     # results[k] = cross_val_score(v, x, y, cv=10, n_jobs=-1,
#     #                              scoring=make_scorer(classification_report_with_accuracy_score, kwargs=k))
#     results[k] = cross_val_score(v, x, y, cv=10, n_jobs=-1, scoring='accuracy')
#     # results_normalized[k] = cross_val_score(v, x_normalized, y, cv=10, n_jobs=-1,
#     #                                         scoring=make_scorer(classification_report_with_accuracy_score, kwargs=k))
#     results_normalized[k] = cross_val_score(v, x_normalized, y, cv=10, n_jobs=-1, scoring='accuracy')
#
#
# print("Results using non-normalized input")
# print_results(results)
#
# print("Results using angular kernel")
# print_results(results_normalized)
