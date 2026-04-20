"""
SQLite evidence archive for citation tool calls.
Used by the witness agent for: search_record, retrieve_document, flag_inconsistency.
The witness answers from its persona — this is for citation/verification only.
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path("data/dossier.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            domain TEXT,
            doc_type TEXT,
            title TEXT,
            authored_date DATE,
            received_date DATE,
            version TEXT,
            source_url TEXT,
            cluster INT,
            topic_tags TEXT,
            body_text TEXT,
            key_claims TEXT,
            quantitative_fields TEXT,
            supersedes TEXT,
            cited_by TEXT,
            available_from DATE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS distortions (
            distortion_id TEXT PRIMARY KEY,
            source_doc_id TEXT,
            distortion_type TEXT,
            original_claim TEXT,
            distorted_claim TEXT,
            detection_evidence TEXT,
            difficulty INT,
            requires_docs TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS information_states (
            episode_id TEXT,
            turn_number INT,
            docs_retrieved TEXT,
            docs_available TEXT,
            claims_made TEXT
        )
    """)
    conn.commit()
    conn.close()


def search_record(query: str, domain: str = None) -> List[Dict]:
    """Tool call: searches evidence archive by query. Returns ranked fact cards."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    sql = "SELECT doc_id, title, key_claims, quantitative_fields FROM documents WHERE body_text LIKE ?"
    params = [f"%{query}%"]
    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    sql += " LIMIT 5"
    results = [dict(r) for r in c.execute(sql, params).fetchall()]
    conn.close()
    return results


def retrieve_document(doc_id: str, version: str = None) -> Optional[Dict]:
    """Tool call: retrieves a specific document by ID, optionally at a specific version."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    sql = "SELECT * FROM documents WHERE doc_id = ?"
    params = [doc_id]
    if version:
        sql += " AND version = ?"
        params.append(version)
    row = c.execute(sql, params).fetchone()
    conn.close()
    return dict(row) if row else None


def flag_inconsistency(claim: str, evidence_id: str) -> Dict:
    """Tool call: formally lodges a dispute linking a claim to a document that refutes it."""
    doc = retrieve_document(evidence_id)
    return {
        "status": "flagged",
        "claim": claim,
        "evidence_id": evidence_id,
        "evidence_title": doc.get("title") if doc else "Document not found",
    }
