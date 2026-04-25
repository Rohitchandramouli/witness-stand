"""Inspect dossier DB quality after build.

Run:
    python scripts/01_inspect_dossier.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from constants import DB_PATH, PERSONAS_DIR  # noqa: E402


DOMAINS = ["financial", "medical", "safety", "technical"]


def _safe_json(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default


def inspect_domain(cur: sqlite3.Cursor, domain: str) -> dict[str, Any]:
    docs = cur.execute(
        """
        SELECT doc_id, title, body_text, key_claims
        FROM documents
        WHERE domain = ?
        ORDER BY doc_id
        """,
        (domain,),
    ).fetchall()

    distortions = cur.execute(
        """
        SELECT distortion_id, difficulty, distortion_type, original_claim, distorted_claim
        FROM distortions
        WHERE domain = ?
        ORDER BY difficulty, distortion_id
        """,
        (domain,),
    ).fetchall()

    persona_path = PERSONAS_DIR / f"{domain}.json"

    total_claims = 0
    empty_claim_docs: list[str] = []

    for doc in docs:
        claims = _safe_json(doc["key_claims"], [])
        total_claims += len(claims)
        if not claims:
            empty_claim_docs.append(doc["doc_id"])

    status = "PASS"
    warnings: list[str] = []

    if len(docs) == 0:
        status = "FAIL"
        warnings.append("No documents inserted.")
    if len(distortions) == 0:
        status = "FAIL"
        warnings.append("No distortions generated.")
    if not persona_path.exists():
        status = "FAIL"
        warnings.append("Persona JSON missing.")
    if total_claims < 5:
        status = "WEAK" if status == "PASS" else status
        warnings.append("Low key-claim count.")

    return {
        "domain": domain,
        "status": status,
        "documents": len(docs),
        "distortions": len(distortions),
        "persona_exists": persona_path.exists(),
        "total_key_claims": total_claims,
        "empty_key_claim_docs": empty_claim_docs,
        "sample_document": _sample_document(docs),
        "sample_distortion": _sample_distortion(distortions),
        "warnings": warnings,
    }


def _sample_document(docs: list[sqlite3.Row]) -> dict[str, Any] | None:
    if not docs:
        return None

    doc = docs[0]
    claims = _safe_json(doc["key_claims"], [])

    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "body_chars": len(doc["body_text"] or ""),
        "sample_key_claims": claims[:2],
    }


def _sample_distortion(distortions: list[sqlite3.Row]) -> dict[str, Any] | None:
    if not distortions:
        return None

    d = distortions[0]
    return {
        "distortion_id": d["distortion_id"],
        "type": d["distortion_type"],
        "difficulty": d["difficulty"],
        "original": (d["original_claim"] or "")[:180],
        "distorted": (d["distorted_claim"] or "")[:180],
    }


def print_report(results: list[dict[str, Any]]) -> None:
    print("\n=== DOSSIER INSPECTION ===")

    for result in results:
        print(f"\n--- {result['domain'].upper()} [{result['status']}] ---")
        print(f"Documents inserted      : {result['documents']}")
        print(f"Distortions generated   : {result['distortions']}")
        print(f"Persona JSON exists     : {result['persona_exists']}")
        print(f"Total key claims        : {result['total_key_claims']}")
        print(f"Docs with empty claims  : {result['empty_key_claim_docs']}")

        if result["sample_document"]:
            doc = result["sample_document"]
            print("\nSample document:")
            print(f"  ID          : {doc['doc_id']}")
            print(f"  Title       : {doc['title']}")
            print(f"  Body chars  : {doc['body_chars']}")
            print(f"  Key claims  : {doc['sample_key_claims']}")

        if result["sample_distortion"]:
            dist = result["sample_distortion"]
            print("\nSample distortion:")
            print(f"  Type        : {dist['type']}")
            print(f"  Difficulty  : {dist['difficulty']}")
            print(f"  Original    : {dist['original']}")
            print(f"  Distorted   : {dist['distorted']}")

        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")

    failed = [r for r in results if r["status"] == "FAIL"]
    weak = [r for r in results if r["status"] == "WEAK"]

    print("\n=== SUMMARY ===")
    print(f"Domains checked : {len(results)}")
    print(f"Passed          : {sum(r['status'] == 'PASS' for r in results)}")
    print(f"Weak            : {len(weak)}")
    print(f"Failed          : {len(failed)}")

    if failed:
        raise SystemExit(1)


def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: dossier DB not found at {DB_PATH}")
        print("Run: python scripts/00_build_dossier.py")
        raise SystemExit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        results = [inspect_domain(conn.cursor(), domain) for domain in DOMAINS]
    finally:
        conn.close()

    print_report(results)


if __name__ == "__main__":
    main()
