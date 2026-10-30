import json
import math
import unittest
from unittest.mock import patch

from src.classification import NaiveBayesClassifier
from src.corpus import ROOT, load_documents
from src.search import SearchIndex
from src.vectorizer import TfidfVectorizer
from main import QUERIES, train_classifier


class SystemTests(unittest.TestCase):
    def test_hand_calculation(self):
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform(["tablet tablet reset", "tablet loan"])[0]
        self.assertAlmostEqual(vectorizer.idf["tablet"], 1)
        self.assertAlmostEqual(vectorizer.idf["reset"], math.log(1.5) + 1)
        self.assertAlmostEqual(vector["tablet"], 2 / math.sqrt(4 + (math.log(1.5) + 1)**2))
        self.assertAlmostEqual(sum(value**2 for value in vector.values()), 1)

    def test_manifest_and_split(self):
        docs = load_documents()
        train = {doc["id"] for doc in docs if doc["split"] == "train"}
        test = {doc["id"] for doc in docs if doc["split"] == "test"}
        self.assertEqual((len(docs), len(train), len(test)), (18, 12, 6))
        self.assertFalse(train & test)
        manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        manifest[0]["path"] = "../outside.txt"
        with patch("src.corpus.Path.read_text", return_value=json.dumps(manifest)):
            with self.assertRaisesRegex(ValueError, "inside data/documents"):
                load_documents()

    def test_no_test_vocabulary_leak(self):
        docs = load_documents()
        augmented = [{**doc, "text": doc["text"] + (" uniquetesttoken" if doc["split"] == "test" else "")}
                     for doc in docs]
        vectorizer, model = train_classifier(augmented)
        self.assertNotIn("uniquetesttoken", vectorizer.idf)
        self.assertEqual(vectorizer.transform(["qzxv blorf"]), [{}])
        self.assertEqual(model.predict_one({}), "Unclassified")

    def test_separable_nb(self):
        # Independent toy example with hand-computed alpha=1 term likelihoods.
        model = NaiveBayesClassifier().fit([{"apple": 2}, {"pear": 2}], ["A", "B"], ["apple", "pear"])
        self.assertAlmostEqual(model.log_terms["A"]["apple"], math.log(3 / 4))
        self.assertAlmostEqual(model.log_terms["A"]["pear"], math.log(1 / 4))
        self.assertEqual(model.predict([{"apple": 1}, {"pear": 1}]), ["A", "B"])

    def test_retrieval_and_empty_inputs(self):
        index = SearchIndex(load_documents())
        for query, expected in zip(QUERIES, ["D01", "G01"]):
            matches = index.search(query)
            self.assertEqual(matches[0]["id"], expected)
            self.assertTrue(all(0 < row["score"] <= 1.000000001 for row in matches))
            self.assertEqual(matches, sorted(matches, key=lambda row: (-row["score"], row["id"])))
        self.assertEqual(index.search(""), [])
        self.assertEqual(index.search("qzxv blorf"), [])
        self.assertEqual(index.search("tablet", limit=0), [])


if __name__ == "__main__":
    unittest.main()
