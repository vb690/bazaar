from modules.basic_supervised import Knn
from modules.basic_supervised import LogReg
from modules.basic_supervised import GaussianNaiveBayes

from modules.toolbox.tester import test_classification

if __name__ == '__main__':

    MODELS = {
        'K-nearest Neighbour': Knn(),
        'Logistic Regression': LogReg(),
        'Gaussian Naive Bayes': GaussianNaiveBayes()
    }

    test_classification(MODELS)
