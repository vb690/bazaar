from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from .scorer import Scorer


def test_classification(models):
    """Function for testing a set of models on a classification task.

        Args:
            - models: is a dictionary where keys are model names and values
                are instatiated models.

        Returns:
            - None
    """
    for name, model in models.items():

        X, y = make_classification(
            n_samples=10000,
            n_features=10,
            n_redundant=4
        )

        for train_i, test_i in StratifiedShuffleSplit().split(X, y):

            X_train, y_train = X[train_i], y[train_i]
            X_test, y_test = X[test_i], y[test_i]

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        scorer = Scorer(predicted, y_test)
        print(f'Model {name}')
        print('')
        print(f'F1 score: {scorer.f1_score()}')
        print(f'Accuracy: {scorer.accuracy()}')
        print(f'Precision: {scorer.precision()}')
        print(f'Recall: {scorer.recall()}')
        print('')

    return None
