"""Count-based TF-IDF with smoothed IDF and L2 normalization."""

import math
import re
from collections import Counter


STOPWORDS = frozenset("a an and are as at be before by for from how in is it of on or that the this to use what with".split())


def tokenize(text):
    return [word for word in re.findall(r"[a-z]{2,}", text.lower()) if word not in STOPWORDS]


class TfidfVectorizer:
    def fit(self, texts):
        texts = list(texts)
        if not texts:
            raise ValueError("At least one training document is required")
        frequency = Counter()
        for text in texts:
            frequency.update(set(tokenize(text)))
        self.idf = {term: math.log((1 + len(texts)) / (1 + count)) + 1
                    for term, count in sorted(frequency.items())}
        return self

    def transform(self, texts):
        if not hasattr(self, "idf"):
            raise ValueError("Fit the vectorizer before transforming text")
        vectors = []
        for text in texts:
            weights = {term: count * self.idf[term]
                       for term, count in Counter(tokenize(text)).items() if term in self.idf}
            norm = math.sqrt(sum(weight * weight for weight in weights.values()))
            vectors.append({term: weight / norm for term, weight in weights.items()} if norm else {})
        return vectors

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)


def cosine_similarity(left, right):
    """Dot product of the L2-normalized vectors produced above."""
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())
