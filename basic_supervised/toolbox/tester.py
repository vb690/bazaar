from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from toolbox.scorer import *

def test_classification (model):

    X, y = make_classification(n_samples = 1000, n_features = 10, n_redundant=4)

    for train_i, test_i in StratifiedShuffleSplit().split(X, y):

        X_train, y_train = X[train_i], y[train_i]
        X_test, y_test = X[test_i], y[test_i]

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    scorer = Scorer(predicted, y_test)
    print('F1 score: {}'.format(scorer.f1_score()))
    print('Accuracy: {}'.format(scorer.accuracy()))
    print('Precision: {}'.format(scorer.precision()))
    print('Recall: {}'.format(scorer.recall()))
