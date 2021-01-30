# Exercises for Learning how to Teach to the Machines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
A repository for going through the basics of machine learning using the book 'Machine Learning In Action'. All the models employs scikit-learn like methods, like `fit` and `predict`, for training and performing the prediction.

## Covered

### Tester
Utility for testing the performance of a model, it has acess to the following metrics via a `scorer` object:
- Recall
- Specificity
- Precision
- False negative rate
- False positive rate
- False discovery rate
- False omission rate
- Accuracy
- F1 score

### Logistic Regression
Simple logistic model with two additional methods for:
- Visualizing the logistic function
- Visualizing the decision boundary in case the number of input features is equal to 2

### K Nearest Neighbours
K nearest neighbour classifier with the following distance metrics availables:
  - Euclidian
  - Manhattan
  - Camberra
  - Chebychev

### Gaussian Naive Bayes
Naive bayes assuming the joint probability being computed through the probaility density function of a gaussian distribution

### Bernulli Naive Bayes
Naive Bayes assuming the features being binary/bolean variables

## References

* https://www.manning.com/books/machine-learning-in-action
* https://en.wikipedia.org/wiki/F1_score
* http://www.molmine.com/magma/analysis/distance.htm
* https://github.com/odubno/GaussNaiveBayes

