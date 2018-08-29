class Scorer():
    def __init__(self, predicted, ground_truth):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.false_positives = sum([True if p != g and p == 1 else False for p, g in zip(predicted, ground_truth)])
        self.false_negatives = sum([True if p != g and p == 0 else False for p, g in zip(predicted, ground_truth)])
        self.true_positives = sum([True if p == g and p == 1 else False for p, g in zip(predicted, ground_truth)])
        self.true_negatives = sum([True if p == g and p == 0 else False for p, g in zip(predicted, ground_truth)])

    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

    def specificity(self):
        return self.true_negatives / (self.true_negatives + self.false_positives)

    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def false_negative_rate(self):
        return self.false_negatives / (self.false_negatives + self.true_positives)

    def false_positive_rate(self):
        return self.false_positives / (self.false_positives + self.true_negatives)

    def false_discovery_rate(self):
        return self.false_positives / (self.false_positives + self.true_positives)

    def false_omission_rate(self):
        return self.false_negatives / (self.false_negatives + self.true_negatives)

    def accuracy(self):
        return (self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)

    def f1_score(self):
        return 2 * ((self.precision() * self.recall()) / (self.precision() + self.recall()))
