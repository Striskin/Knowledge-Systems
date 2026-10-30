"""Run classification and both required natural-language search examples."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from src.classification import NaiveBayesClassifier
from src.corpus import CATEGORIES, ROOT, load_documents
from src.search import SearchIndex
from src.vectorizer import TfidfVectorizer


QUERIES = [
    "How do I restart a frozen tablet before a factory reset?",
    "What approval and return records are needed for a device loan?",
]


def train_classifier(documents):
    training = [doc for doc in documents if doc["split"] == "train"]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(doc["text"] for doc in training)
    model = NaiveBayesClassifier().fit(vectors, [doc["category"] for doc in training], vectorizer.idf)
    return vectorizer, model


def run(queries=None):
    documents = load_documents()
    vectorizer, model = train_classifier(documents)
    testing = [doc for doc in documents if doc["split"] == "test"]
    predictions = model.predict(vectorizer.transform(doc["text"] for doc in testing))
    rows = [{"id": doc["id"], "title": doc["title"], "text": doc["text"],
             "expected": doc["category"], "predicted": predicted}
            for doc, predicted in zip(testing, predictions)]
    confusion = {actual: {predicted: 0 for predicted in CATEGORIES} for actual in CATEGORIES}
    for row in rows:
        confusion[row["expected"]].setdefault(row["predicted"], 0)
        confusion[row["expected"]][row["predicted"]] += 1
    index = SearchIndex(documents)
    paths = sorted({*ROOT.glob("src/*.py"), ROOT / "main.py", ROOT / "practice.py",
                    *ROOT.glob("tests/*.py"), ROOT / "data/manifest.json",
                    *(ROOT / doc["path"] for doc in documents)})
    return {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "document_count": len(documents), "train_count": len(documents) - len(testing),
        "test_count": len(testing), "training_vocabulary_size": len(vectorizer.idf),
        "accuracy": sum(row["expected"] == row["predicted"] for row in rows) / len(rows),
        "classification": rows, "confusion_matrix": confusion,
        "searches": [{"query": query, "results": index.search(query)} for query in (queries or QUERIES)],
        "unknown_query_results": index.search("qzxv blorf"),
        "blank_classification": model.predict_one(vectorizer.transform([""])[0]),
        "sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths},
    }


def transcript(result):
    lines = ["LUMENBRIDGE | KNOWLEDGE MANAGEMENT PROTOTYPE",
             f"Executed UTC: {result['executed_utc']}",
             f"Python {result['python']} | {result['document_count']} synthetic documents | "
             f"{result['train_count']} train / {result['test_count']} test",
             f"Training vocabulary: {result['training_vocabulary_size']} | Held-out accuracy: {result['accuracy']:.3f}",
             "", "CLASSIFICATION (expected -> predicted)"]
    lines.extend(f"{row['id']}: {row['expected']} -> {row['predicted']}" for row in result["classification"])
    for search in result["searches"]:
        lines.extend(["", f"QUERY: {search['query']}"])
        lines.extend(f"{row['id']} | {row['score']:.4f} | {row['title']}" for row in search["results"])
        if not search["results"]:
            lines.append("No matching vocabulary; no documents retrieved.")
    lines.extend(["", f"Unknown query qzxv blorf: {result['unknown_query_results']}",
                  f"Blank classification: {result['blank_classification']}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Run a custom query instead of the two demonstration queries")
    parser.add_argument("--classify-text", help="Classify an additional document body")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evidence")
    args = parser.parse_args()
    result = run([args.query] if args.query is not None else None)
    if args.classify_text is not None:
        vectorizer, model = train_classifier(load_documents())
        result["custom_classification"] = {"text": args.classify_text,
            "predicted": model.predict_one(vectorizer.transform([args.classify_text])[0])}
    output = transcript(result)
    if "custom_classification" in result:
        output += f"\nCUSTOM CLASSIFICATION: {result['custom_classification']['predicted']}\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "execution.txt").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
