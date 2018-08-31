from toolbox.tester import *
from logistic_regression import *
from k_nearest_neighbour import *

if __name__ == '__main__':

    test_classification(Knn(k=3))
