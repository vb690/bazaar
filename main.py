from modules.basic_supervised import Knn
from modules.basic_supervised import LogReg
from modules.basic_supervised import GaussianNaiveBayes

from modules.toolbox.tester import test_classification

if __name__ == '__main__':

    test_classification(Knn())
    test_classification(LogReg())
    test_classification(GaussianNaiveBayes())
