#!/usr/bin/env python3 


import numpy as np
import pprint
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from plot_learning_curve import plot_learning_curve

with open("spambase_clean.data") as f:
    data = np.loadtxt(f, delimiter=',')
    np.random.shuffle(data)  # mettere nel report che migliora le performance
    x = data[:, 0:54]
    y = data[:, 57]

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

print("Number of mislabeled points out of a total {} points : {}".format(x.shape[0], (y != y_pred).sum()))

# Cross Validation
r = cross_val_score(gaussian_bayes, x, y, cv=10, n_jobs=-1, scoring='accuracy')
print("Minimum accuracy for Gaussian Naive Bayes: {}".format(r.min()))
print("Average accuracy for Gaussian Naive Bayes: {}".format(r.mean()))
print("Maximum accuracy for Gaussian Naive Bayes: {}".format(r.max()))

pp = pprint.PrettyPrinter(indent=2)
# Precision, Recall, F1-score, support
pp.pprint(classification_report(y, y_pred, output_dict=True))
