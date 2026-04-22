"""
SQLite evidence archive for citation tool calls.
Used by the witness agent for: search_record, retrieve_document, flag_inconsistency.
The witness answers from its persona — this is for citation/verification only.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from constants import DB_PATH


def init_db() -> None:
    """Creates all tables if they don't exist. Safe to call multiple times."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id          TEXT PRIMARY KEY,
            domain          TEXT NOT NULL,
            doc_type        TEXT NOT NULL,
            title           TEXT NOT NULL,
            authored_date   DATE,
            received_date   DATE,
            version         TEXT DEFAULT 'v1.0',
            source_url      TEXT,
            cluster         INTEGER,
            topic_tags      TEXT,
            body_text       TEXT,
            key_claims      TEXT,
            quantitative_fields TEXT,
            supersedes      TEXT,
            cited_by        TEXT,
            available_from  DATE
        )
    """)

    c.execute("""
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
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS information_states (
            episode_id      TEXT NOT NULL,
            turn_number     INTEGER NOT NULL,
            docs_retrieved  TEXT NOT NULL DEFAULT '[]',
            docs_available  TEXT NOT NULL DEFAULT '[]',
            claims_made     TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (episode_id, turn_number)
        )
    """)

    conn.commit()
    conn.close()


def search_record(query: str, domain: Optional[str] = None) -> List[Dict]:
    """
    Tool call: full-text search across documents.
    Returns up to 5 matching records as fact cards.
    Used by the witness to verify an external claim before contesting it.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    sql = """
        SELECT doc_id, domain, title, key_claims, quantitative_fields, available_from
        FROM documents
        WHERE body_text LIKE ?
    """
    params: List = [f"%{query}%"]

    if domain:
        sql += " AND domain = ?"
        params.append(domain)

    sql += " LIMIT 5"

    rows = c.execute(sql, params).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def retrieve_document(doc_id: str, version: Optional[str] = None) -> Optional[Dict]:
    """
    Tool call: retrieves a specific document by ID for formal citation.
    The witness already knows what this document says — retrieval is a
    citation action, not a lookup. Optionally filters to a specific version.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    sql = "SELECT * FROM documents WHERE doc_id = ?"
    params: List = [doc_id]

    if version:
        sql += " AND version = ?"
        params.append(version)

    row = c.execute(sql, params).fetchone()
    conn.close()

    return dict(row) if row else None


def flag_inconsistency(claim: str, evidence_id: str) -> Dict:
    """
    Tool call: formally lodges a dispute against a specific claim,
    linking it to a document that refutes it. Changes the Questioner's
    available attack surface — flagged claims narrow what it can assert.
    """
    doc = retrieve_document(evidence_id)
    return {
        "status": "flagged",
        "claim": claim,
        "evidence_id": evidence_id,
        "evidence_title": doc.get("title") if doc else "Document not found",
        "evidence_available_from": doc.get("available_from") if doc else None,
    }


def get_distortions_for_domain(domain: str, difficulty: List[int]) -> List[Dict]:
    """
    Returns distortion templates for a given domain and difficulty range.
    Called by the panel scheduler when loading the injection pool for a task.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    placeholders = ",".join("?" * len(difficulty))
    sql = f"""
        SELECT * FROM distortions
        WHERE domain = ?
        AND difficulty IN ({placeholders})
    """
    rows = c.execute(sql, [domain] + difficulty).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def log_information_state(
    episode_id: str,
    turn_number: int,
    docs_retrieved: List[str],
    docs_available: List[str],
    claims_made: List[str],
) -> None:
    """
    Writes the witness's information state at a given turn.
    Called by environment.py after each step.
    Read by episode_grader.py for audit trail verification.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO information_states
        (episode_id, turn_number, docs_retrieved, docs_available, claims_made)
        VALUES (?, ?, ?, ?, ?)
    """, (
        episode_id,
        turn_number,
        json.dumps(docs_retrieved),
        json.dumps(docs_available),
        json.dumps(claims_made),
    ))
    conn.commit()
    conn.close()


def get_information_state(episode_id: str, turn_number: int) -> Optional[Dict]:
    """
    Retrieves the witness's information state at a specific turn.
    Used by episode_grader.py to check temporal consistency:
    did the witness cite evidence that was available at that turn?
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    row = c.execute("""
        SELECT * FROM information_states
        WHERE episode_id = ? AND turn_number = ?
    """, (episode_id, turn_number)).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "episode_id": row["episode_id"],
        "turn_number": row["turn_number"],
        "docs_retrieved": json.loads(row["docs_retrieved"]),
        "docs_available": json.loads(row["docs_available"]),
        "claims_made": json.loads(row["claims_made"]),
    }