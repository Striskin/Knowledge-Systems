"""TF-IDF cosine retrieval over all documents in the collection."""

from src.vectorizer import TfidfVectorizer, cosine_similarity


class SearchIndex:
    def __init__(self, documents):
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(doc["text"] for doc in self.documents)

    def search(self, query, limit=3):
        if limit <= 0:
            return []
        vector = self.vectorizer.transform([query])[0]
        matches = []
        for doc, candidate in zip(self.documents, self.vectors):
            score = cosine_similarity(vector, candidate)
            if score > 0:
                matches.append({"id": doc["id"], "title": doc["title"],
                                "category": doc["category"], "score": score})
        return sorted(matches, key=lambda item: (-item["score"], item["id"]))[:limit]
