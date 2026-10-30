"""Multinomial Naive Bayes accepting nonnegative TF-IDF weights."""

import math
from collections import Counter


class NaiveBayesClassifier:
    def fit(self, vectors, labels, vocabulary):
        if not vectors or len(vectors) != len(labels) or not vocabulary:
            raise ValueError("Provide aligned nonempty training vectors, labels and vocabulary")
        self.vocabulary = set(vocabulary)
        counts = Counter(labels)
        self.classes = sorted(counts)
        self.log_prior = {label: math.log(counts[label] / len(labels)) for label in self.classes}
        self.log_terms = {}
        for label in self.classes:
            totals = Counter()
            for vector, actual in zip(vectors, labels):
                if actual == label:
                    totals.update(vector)
            denominator = sum(totals.values()) + len(self.vocabulary)
            self.log_terms[label] = {term: math.log((totals[term] + 1) / denominator)
                                     for term in self.vocabulary}
        return self

    def predict_one(self, vector):
        if not hasattr(self, "classes"):
            raise ValueError("Train the classifier before predicting")
        known = {term: value for term, value in vector.items()
                 if term in self.vocabulary and value > 0}
        if not known:
            return "Unclassified"
        return max(self.classes, key=lambda label: self.log_prior[label] + sum(
            weight * self.log_terms[label][term] for term, weight in known.items()))

    def predict(self, vectors):
        return [self.predict_one(vector) for vector in vectors]
