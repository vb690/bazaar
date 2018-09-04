import numpy as np

class GaussianNaiveBayes():

    def __init__(self):
        pass

    @staticmethod
    def group_by(values_array, grouping_array):
        grouped = {group : [] for group in np.unique(grouping_array)}
        for values, group in zip(values_array, grouping_array):

            grouped[group].append(values)

        grouped = {group : np.array(values) for group, values in grouped.items()}
        return grouped

    @staticmethod
    def compute_statistics(grouped):
        labels_stats = {}
        for label, values in grouped.items():

            labels_stats[label] = {'means' : values.mean(axis=0), 'stds' : values.std(axis=0)}

        return labels_stats

    def _compute_probabilities(self):
        labels, occurences = np.unique(self.y_train, return_counts=True)
        return {label : occurence/len(self.y_train) for label, occurence in zip(labels, occurences)}

    def fit(self, X_train, y_train, labels_probabilities = None):
        setattr(self, 'X_train', X_train)
        setattr(self, 'y_train', y_train)
        if labels_probabilities != None:
            setattr(self, 'labels_probabilities', labels_probabilities)
        else:
            setattr(self, 'labels_probabilities', self._compute_probabilities())
        grouped = self.group_by(self.X_train, self.y_train)
        setattr(self, 'labels_stats', self.compute_statistics(grouped))
