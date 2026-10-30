"""Load the manifest and the synthetic UTF-8 document bodies."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("Device Support", "Governance", "Inclusive Teaching")


def load_documents(root=ROOT):
    root = Path(root).resolve()
    records = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    documents, seen = [], set()
    for record in records:
        required = {"id", "title", "category", "owner", "version", "review_date", "path", "split"}
        if not required <= record.keys():
            raise ValueError("A manifest record is missing required metadata")
        if record["id"] in seen or record["split"] not in {"train", "test"}:
            raise ValueError("Duplicate document ID or invalid experimental split")
        if record["category"] not in CATEGORIES:
            raise ValueError("Unknown document category")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root / "data/documents"):
            raise ValueError("Document paths must stay inside data/documents")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty document: {record['id']}")
        seen.add(record["id"])
        documents.append({**record, "text": text})
    if not documents or not {"train", "test"} <= {doc["split"] for doc in documents}:
        raise ValueError("The corpus must contain both training and test documents")
    if {doc["category"] for doc in documents if doc["split"] == "train"} != set(CATEGORIES):
        raise ValueError("Every category needs training documents")
    return documents
