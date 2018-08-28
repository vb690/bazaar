from logistic_regression import *

def test_classification (model):

    X, y = make_classification(n_samples = 1000, n_features = 5)

    for train_i, test_i in StratifiedShuffleSplit().split(X, y):

        X_train, y_train = X[train_i], y[train_i]
        X_test, y_test = X[test_i], y[test_i]

    model.fit(X_train, y_train)
    model.predict(X_test, y_test)
    model.visualize_decision_function(X_test)
    model.visualize_decision_boundary(X_test, y_test)


if __name__ == '__main__':

    test_classification(LogReg())
