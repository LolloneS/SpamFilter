#!/usr/bin/env python3 


import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from plot_learning_curve import plot_learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

with open("spambase_clean.data") as f:
    data = np.loadtxt(f, delimiter=',')
    np.random.shuffle(data)  # mettere nel report che migliora le performance
    x = data[:, 0:54]
    y = data[:, 57]
    y = y.astype(int)

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# Plot learning curves
title = "Learning Curves (Naive Bayes)"
# plot = plot_learning_curve(gaussian_bayes, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
# plot.show()


# Naive Bayes classification using gaussian
gaussian_bayes = GaussianNB()
y_pred = gaussian_bayes.fit(x, y).predict(x)
y_pred = y_pred.astype(int)

print("Number of mislabeled points out of a total {} points : {}".format(x.shape[0], (y != y_pred).sum()))

# Cross Validation
r = cross_val_score(gaussian_bayes, x, y, cv=10, n_jobs=-1, scoring='accuracy')
print("Minimum accuracy for Gaussian Naive Bayes: {}".format(r.min()))
print("Average accuracy for Gaussian Naive Bayes: {}".format(r.mean()))
print("Maximum accuracy for Gaussian Naive Bayes: {}".format(r.max()))


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

print("Confusion matrix:\n")
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
(tn, fp, fn, tp)
plot_confusion_matrix(y, y_pred, classes=y_pred, normalize=True, title="Normalized Confusion matrix")
plt.show()

pp = pprint.PrettyPrinter(indent=2)
# Precision, Recall, F1-score, support
pp.pprint(classification_report(y, y_pred, output_dict=True))
