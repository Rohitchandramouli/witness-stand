"""
SQLite evidence archive for citation, distortion, and information-state tracking.
This module is intentionally deterministic and lightweight for RL training.
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from constants import DB_PATH


JsonDict = Dict[str, Any]


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _loads_json(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def init_db() -> None:
    """Creates all dossier tables and indexes. Safe to call repeatedly."""
    with _connect() as conn:
        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id              TEXT PRIMARY KEY,
                domain              TEXT NOT NULL,
                doc_type            TEXT NOT NULL,
                title               TEXT NOT NULL,
                authored_date       DATE,
                received_date       DATE,
                version             TEXT DEFAULT 'v1.0',
                source_url          TEXT,
                cluster             INTEGER,
                topic_tags          TEXT,
                body_text           TEXT,
                key_claims          TEXT,
                quantitative_fields TEXT,
                supersedes          TEXT,
                cited_by            TEXT,
                available_from      DATE
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS distortions (
                distortion_id       TEXT PRIMARY KEY,
                source_doc_id       TEXT NOT NULL,
                domain              TEXT NOT NULL,
                distortion_type     TEXT NOT NULL,
                original_claim      TEXT NOT NULL,
                distorted_claim     TEXT NOT NULL,
                detection_evidence  TEXT NOT NULL,
                difficulty          INTEGER NOT NULL,
                requires_docs       TEXT NOT NULL,
                FOREIGN KEY (source_doc_id) REFERENCES documents (doc_id)
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS information_states (
                episode_id      TEXT NOT NULL,
                turn_number     INTEGER NOT NULL,
                docs_retrieved  TEXT NOT NULL DEFAULT '[]',
                docs_available  TEXT NOT NULL DEFAULT '[]',
                claims_made     TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (episode_id, turn_number)
            )
            """
        )

        c.execute("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_documents_available ON documents(available_from)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_distortions_domain ON distortions(domain)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_distortions_difficulty ON distortions(difficulty)")


def search_record(query: str, domain: Optional[str] = None, limit: int = 5) -> List[JsonDict]:
    """
    Searches dossier documents and returns compact fact cards.

    Empty query intentionally returns domain documents. This is useful for
    retrieving genuine evidence pools during environment setup.
    """
    safe_limit = max(1, min(limit, 25))
    query = (query or "").strip()

    sql = """
        SELECT
            doc_id,
            domain,
            title,
            doc_type,
            source_url,
            key_claims,
            quantitative_fields,
            available_from
        FROM documents
        WHERE 1 = 1
    """
    params: List[Any] = []

    if query:
        sql += " AND (body_text LIKE ? OR title LIKE ? OR key_claims LIKE ?)"
        like_query = f"%{query}%"
        params.extend([like_query, like_query, like_query])

    if domain:
        sql += " AND domain = ?"
        params.append(domain)

    sql += " ORDER BY available_from DESC, doc_id ASC LIMIT ?"
    params.append(safe_limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    results: List[JsonDict] = []
    for row in rows:
        item = dict(row)
        item["key_claims"] = _loads_json(item.get("key_claims"), [])
        item["quantitative_fields"] = _loads_json(item.get("quantitative_fields"), {})
        results.append(item)

    return results


def retrieve_document(doc_id: str, version: Optional[str] = None) -> Optional[JsonDict]:
    """Retrieves a full document by ID, optionally constrained by version."""
    sql = "SELECT * FROM documents WHERE doc_id = ?"
    params: List[Any] = [doc_id]

    if version:
        sql += " AND version = ?"
        params.append(version)

    with _connect() as conn:
        row = conn.execute(sql, params).fetchone()

    if not row:
        return None

    doc = dict(row)
    doc["key_claims"] = _loads_json(doc.get("key_claims"), [])
    doc["quantitative_fields"] = _loads_json(doc.get("quantitative_fields"), {})
    doc["topic_tags"] = _loads_json(doc.get("topic_tags"), [])
    doc["cited_by"] = _loads_json(doc.get("cited_by"), [])
    return doc


def flag_inconsistency(claim: str, evidence_id: str) -> JsonDict:
    """Creates a structured inconsistency flag linked to a refuting document."""
    doc = retrieve_document(evidence_id)

    return {
        "status": "flagged" if doc else "evidence_not_found",
        "claim": claim,
        "evidence_id": evidence_id,
        "evidence_title": doc.get("title") if doc else None,
        "evidence_available_from": doc.get("available_from") if doc else None,
    }


def get_distortions_for_domain(domain: str, difficulty: List[int]) -> List[JsonDict]:
    """Returns distortion templates for a domain and difficulty set."""
    if not difficulty:
        return []

    clean_difficulty = sorted({int(level) for level in difficulty if int(level) > 0})
    if not clean_difficulty:
        return []

    placeholders = ",".join("?" for _ in clean_difficulty)
    sql = f"""
        SELECT *
        FROM distortions
        WHERE domain = ?
          AND difficulty IN ({placeholders})
        ORDER BY difficulty ASC, distortion_id ASC
    """

    with _connect() as conn:
        rows = conn.execute(sql, [domain, *clean_difficulty]).fetchall()

    results: List[JsonDict] = []
    for row in rows:
        item = dict(row)
        item["requires_docs"] = _loads_json(item.get("requires_docs"), [])
        results.append(item)

    return results


def log_information_state(
    episode_id: str,
    turn_number: int,
    docs_retrieved: List[str],
    docs_available: List[str],
    claims_made: List[str],
) -> None:
    """Persists what the witness could know at a given turn."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO information_states
            (episode_id, turn_number, docs_retrieved, docs_available, claims_made)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                turn_number,
                json.dumps(docs_retrieved),
                json.dumps(docs_available),
                json.dumps(claims_made),
            ),
        )


def get_information_state(episode_id: str, turn_number: int) -> Optional[JsonDict]:
    """Retrieves the witness information state for one episode turn."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM information_states
            WHERE episode_id = ?
              AND turn_number = ?
            """,
            (episode_id, turn_number),
        ).fetchone()

    if not row:
        return None

    return {
        "episode_id": row["episode_id"],
        "turn_number": row["turn_number"],
        "docs_retrieved": _loads_json(row["docs_retrieved"], []),
        "docs_available": _loads_json(row["docs_available"], []),
        "claims_made": _loads_json(row["claims_made"], []),
    }