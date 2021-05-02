

class Scorer:
    def __init__(self, predicted, ground_truth):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.false_positives = sum(
            [True if p != g and p == 1 else False for p, g in
                zip(predicted, ground_truth)
             ]
        )
        self.false_negatives = sum(
            [True if p != g and p == 0 else False for p, g in
                zip(predicted, ground_truth)
             ]
        )
        self.true_positives = sum(
            [True if p == g and p == 1 else False for p, g in
                zip(predicted, ground_truth)
             ]
        )
        self.true_negatives = sum(
            [True if p == g and p == 0 else False for p, g in
                zip(predicted, ground_truth)
             ]
        )

    def recall(self):
        """Compute the recall score.
        """
        nominator = self.true_positives
        denominator = (self.true_positives + self.false_negatives)
        return nominator / denominator

    def specificity(self):
        """Compute the specificity score.
        """
        nominator = self.true_negatives
        denominator = (self.true_negatives + self.false_positives)
        return nominator / denominator

    def precision(self):
        """Compute the precision score.
        """
        nominator = self.true_positives
        denominator = (self.true_positives + self.false_positives)
        return nominator / denominator

    def false_negative_rate(self):
        """Compute false negative rate.
        """
        nominator = self.false_negatives
        denominator = (self.false_negatives + self.true_positives)
        return nominator / denominator

    def false_positive_rate(self):
        """Compute false positive rate.
        """
        nominator = self.false_positives
        denominator = (self.false_positives + self.true_negatives)
        return nominator / denominator

    def false_discovery_rate(self):
        """Compute false discovery rate.
        """
        nominator = self.false_positives
        denominator = (self.false_positives + self.true_positives)
        return nominator / denominator

    def false_omission_rate(self):
        """Compute false omission rate.
        """
        nominator = self.false_negatives
        denominator = (self.false_negatives + self.true_negatives)
        return nominator / denominator

    def accuracy(self):
        """Compute accuracy score.
        """
        nominator = (self.true_positives + self.true_negatives)
        denominator = (
            self.true_positives +
            self.true_negatives +
            self.false_positives +
            self.false_negatives
        )
        return nominator / denominator

    def f1_score(self):
        """Compute f1 score.
        """
        nominator = (self.precision() * self.recall())
        denominator = (self.precision() + self.recall())
        return 2 * (nominator / denominator)
