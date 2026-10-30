"""Six new supporting exercises, distinct from historical coursework."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

from main import QUERIES, train_classifier
from src.corpus import ROOT, load_documents
from src.search import SearchIndex
from src.vectorizer import TfidfVectorizer


TRIPLES = [
    ["Service Desk", "maintains", "Tablet"], ["Learning Design", "designs", "Lesson"],
    ["Lesson", "uses", "Tablet"], ["Lesson", "supports", "Learner"],
    ["Loan", "assigns", "Tablet"], ["Compliance", "approves", "Policy"],
    ["Support Manual", "describes", "Tablet"], ["Teaching Guide", "guides", "Lesson"],
    ["Privacy Policy", "protects", "Learner"], ["Loan Policy", "governs", "Loan"],
]


def main():
    documents = load_documents()
    vectorizer, model = train_classifier(documents)
    example = next(doc for doc in documents if doc["id"] == "D05")
    calculation = TfidfVectorizer()
    vectors = calculation.fit_transform(["tablet tablet reset", "tablet loan"])
    notebooks = []
    for path in sorted(ROOT.glob("*.ipynb")):
        if path.name.startswith("00_"):
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        notebooks.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "code_cells": sum(cell["cell_type"] == "code" for cell in cells),
                          "saved_errors": [{"cell": number, "type": output.get("ename")}
                              for number, cell in enumerate(cells, 1) for output in cell.get("outputs", [])
                              if output.get("output_type") == "error"],
                          "has_source": any("".join(cell.get("source", [])).strip() for cell in cells)})
    result = {"executed_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "New supporting exercises created for this completion. Official course topic numbers were not provided.",
        "knowledge_audit": {"documents": len(documents), "by_owner": dict(Counter(doc["owner"] for doc in documents))},
        "lifecycle_trace": ["Create draft D01", "Store D01.txt and metadata", "Classify Device Support",
                            "Share through search", "Apply restart procedure", "Review feedback for version 1.1"],
        "conceptual_triples": TRIPLES,
        "document_bindings": {"D01": "Support Manual", "T01": "Teaching Guide", "G01": "Loan Policy", "G02": "Privacy Policy"},
        "vectorization": {"idf": calculation.idf, "first_vector": vectors[0],
                          "squared_norm": sum(value * value for value in vectors[0].values())},
        "classification": {"id": example["id"], "expected": example["category"],
                           "predicted": model.predict_one(vectorizer.transform([example["text"]])[0])},
        "retrieval": SearchIndex(documents).search(QUERIES[1]),
        "improvement_proposal": {"id": "D01", "proposed_version": "1.1", "status": "draft only",
             "change": "Obtain Service Desk lead approval before a factory reset."},
        "existing_coursework_inventory": notebooks}
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence/practice.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
