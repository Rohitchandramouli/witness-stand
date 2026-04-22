"""
scripts/build_dossier.py — one-time offline build pipeline.

Fetches real documents from all four domain source URLs, runs a single
LLM extraction pass per domain via Groq, and writes:

  data/personas/<domain>.json     — synthesised persona system prompt
  data/dossier.db                 — SQLite evidence archive with:
                                      • documents table (real source text)
                                      • distortions table (attack templates)
                                      • information_states table (episode audit log)

Run once before any training episode. Safe to re-run — overwrites existing
persona JSONs and upserts into the database.

Usage:
    python scripts/build_dossier.py               # build all four domains
    python scripts/build_dossier.py --domain financial  # build one domain
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from groq import Groq

# ── Make root importable from scripts/ ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dossier import DOSSIER_REGISTRY
from dossier.base import DossierBase
from dossier.dossier_db import init_db
from dossier.persona_builder import build_persona
from constants import DB_PATH, PERSONAS_DIR, WITNESS_MODEL

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 15      # seconds per HTTP fetch
_MAX_SOURCE_CHARS = 6000   # chars of source text fed to LLM per domain
_DISTORTIONS_PER_DOC = 3   # distortion templates generated per document
_DIFFICULTIES = [1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Witness Stand dossiers")
    parser.add_argument(
        "--domain",
        choices=list(DOSSIER_REGISTRY.keys()),
        default=None,
        help="Build a single domain (default: all four)",
    )
    args = parser.parse_args()

    _check_api_key()

    # Ensure directories and DB schema exist
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()

    domains = [args.domain] if args.domain else list(DOSSIER_REGISTRY.keys())

    print(f"\n=== The Witness Stand — Dossier Build ===")
    print(f"Building {len(domains)} domain(s): {domains}")
    print(f"Output: personas → {PERSONAS_DIR}  |  DB → {DB_PATH}\n")

    for domain in domains:
        print(f"── {domain.upper()} ──────────────────────────────────────────")
        dossier_class = DOSSIER_REGISTRY[domain]
        _build_domain(dossier_class)
        print()

    print("=== Build complete ===")
    print(f"Personas: {list(PERSONAS_DIR.glob('*.json'))}")
    print(f"DB size:  {DB_PATH.stat().st_size // 1024} KB")


# ─────────────────────────────────────────────────────────────────────────────
#  PER-DOMAIN BUILD
# ─────────────────────────────────────────────────────────────────────────────

def _build_domain(dossier_class) -> None:
    """
    Full build pipeline for one domain:
      1. Fetch documents from source URLs
      2. Synthesise persona via LLM (build_persona)
      3. Insert documents into dossier.db
      4. Generate distortion templates via LLM
      5. Insert distortions into dossier.db
    """
    dossier: DossierBase = dossier_class()
    domain = dossier.domain

    # Step 1 — fetch raw documents
    print(f"  [1/4] Fetching documents from {len(dossier.source_urls)} sources...")
    raw_docs = _fetch_all_sources(dossier)
    print(f"        Fetched {len(raw_docs)} document(s).")

    # Step 2 — build persona JSON (calls Groq once)
    print(f"  [2/4] Synthesising persona via LLM...")
    build_persona(dossier_class)   # writes data/personas/<domain>.json

    # Step 3 — insert documents into DB
    print(f"  [3/4] Inserting documents into dossier.db...")
    _insert_documents(domain, raw_docs)

    # Step 4 — generate distortion templates
    print(f"  [4/4] Generating distortion templates...")
    distortions = _generate_distortions(domain, raw_docs)
    _insert_distortions(distortions)
    print(f"        Generated {len(distortions)} distortion template(s).")


# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_sources(dossier: DossierBase) -> List[Dict[str, Any]]:
    """
    Fetches all source URLs for a dossier and returns a list of document dicts.
    Each dict has: url, title, body_text, doc_type, authored_date.
    Failures are caught and logged — a missing source reduces richness
    but does not abort the build.
    """
    docs = []
    for url in dossier.source_urls:
        try:
            doc = _fetch_single(url, dossier.domain)
            if doc:
                docs.append(doc)
                print(f"        ✓ {url[:60]}")
        except Exception as e:
            print(f"        ✗ {url[:60]} — {e}")
    return docs


def _fetch_single(url: str, domain: str) -> Dict[str, Any]:
    """
    Routes each URL to a domain-aware fetcher and returns a document dict.
    """
    if "huggingface.co/api/models" in url:
        return _fetch_hf_models(domain)
    elif "clinicaltrials.gov/api/v2" in url:
        return _fetch_clinical_trials()
    elif "eutils.ncbi.nlm.nih.gov" in url:
        return _fetch_pubmed(domain)
    elif "ntsb.gov" in url or "data.ntsb.gov" in url:
        return _fetch_ntsb(url)
    elif "nvlpubs.nist.gov" in url or "nist.gov" in url:
        return _fetch_nist(url)
    else:
        return _fetch_generic(url, domain)


def _fetch_hf_models(domain: str) -> Dict[str, Any]:
    resp = requests.get(
        "https://huggingface.co/api/models",
        params={"sort": "downloads", "direction": -1, "limit": 10, "full": "true"},
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    models = resp.json()
    lines = []
    for m in models[:8]:
        card = m.get("cardData", {})
        lines.append(
            f"Model: {m.get('id', '')} | Downloads: {m.get('downloads', 0)} | "
            f"License: {card.get('license', 'unspecified')} | "
            f"Tags: {', '.join(m.get('tags', [])[:8])}"
        )
    return {
        "url": "https://huggingface.co/api/models",
        "title": "HuggingFace Model Cards — Top Downloaded Models",
        "body_text": "\n".join(lines),
        "doc_type": "dataset",
        "authored_date": "2024-01-01",
    }


def _fetch_clinical_trials() -> Dict[str, Any]:
    resp = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={
            "query.locn": "India",
            "filter.overallStatus": "COMPLETED",
            "fields": (
                "NCTId,BriefTitle,Condition,InterventionName,"
                "PrimaryOutcomeMeasure,Phase,EnrollmentCount,"
                "StartDate,CompletionDate"
            ),
            "pageSize": 10,
            "format": "json",
        },
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    studies = resp.json().get("studies", [])
    lines = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_m = proto.get("identificationModule", {})
        status_m = proto.get("statusModule", {})
        design_m = proto.get("designModule", {})
        lines.append(
            f"Trial: {id_m.get('nctId', '')} | {id_m.get('briefTitle', '')} | "
            f"Phase: {design_m.get('phases', [])} | "
            f"Status: {status_m.get('overallStatus', '')} | "
            f"Completion: {status_m.get('completionDateStruct', {}).get('date', '')}"
        )
    return {
        "url": "https://clinicaltrials.gov/api/v2/studies",
        "title": "ClinicalTrials.gov — Completed Indian Clinical Trials",
        "body_text": "\n".join(lines),
        "doc_type": "dataset",
        "authored_date": "2024-01-01",
    }


def _fetch_pubmed(domain: str) -> Dict[str, Any]:
    domain_queries = {
        "medical":   "clinical trial India adverse events pharmacology",
        "financial": "NBFC risk assessment India financial stability",
        "safety":    "industrial safety incident investigation India",
        "technical": "AI fairness bias audit algorithmic accountability",
    }
    query = domain_queries.get(domain, "AI safety India")

    # Search for PMIDs
    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": 5, "retmode": "json"},
        timeout=_REQUEST_TIMEOUT,
    )
    search.raise_for_status()
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {}

    # Fetch abstracts
    fetch = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"},
        timeout=_REQUEST_TIMEOUT,
    )
    fetch.raise_for_status()
    return {
        "url": "https://eutils.ncbi.nlm.nih.gov",
        "title": f"PubMed Abstracts — {query}",
        "body_text": fetch.text[:3000],
        "doc_type": "paper",
        "authored_date": "2024-01-01",
    }


def _fetch_ntsb(url: str) -> Dict[str, Any]:
    if "data.ntsb.gov" in url:
        return {
            "url": url,
            "title": "NTSB Aviation Accident Database",
            "body_text": (
                "NTSB Aviation Accident Database (avdata): structured accident records "
                "from 1982 to present. Fields: event_id, event_date, location, country, "
                "air_carrier, aircraft_damage, injury_severity, probable_cause, "
                "contributing_factors, weather_condition, flight_phase. "
                "Available at: https://data.ntsb.gov/avdata"
            ),
            "doc_type": "dataset",
            "authored_date": "2024-01-01",
        }
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return {
        "url": url,
        "title": "NTSB Accident Investigation Reports",
        "body_text": resp.text[:3000],
        "doc_type": "report",
        "authored_date": "2024-01-01",
    }


def _fetch_nist(url: str) -> Dict[str, Any]:
    """
    Fetches NIST documents. If the URL returns a PDF (binary),
    extracts real text using PyPDF2 instead of decoding raw bytes.
    Requires: pip install PyPDF2
    """
    import io
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    # Detect PDF by magic bytes, not Content-Type (servers lie)
    if resp.content[:4] == b"%PDF":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(resp.content))
            # Extract first 10 pages — enough for framework terminology
            text = ""
            for i in range(min(10, len(reader.pages))):
                page_text = reader.pages[i].extract_text() or ""
                text += page_text + "\n"
                if len(text) > 5000:
                    break
            if not text.strip():
                raise ValueError("PDF text extraction returned empty")
        except Exception as e:
            raise RuntimeError(
                f"PDF at {url} could not be parsed: {e}. "
                f"Install PyPDF2: pip install PyPDF2"
            )
    else:
        text = resp.text

    return {
        "url": url,
        "title": "NIST AI Risk Management Framework",
        "body_text": text[:5000],
        "doc_type": "regulation",
        "authored_date": "2023-01-26",
    }


def _fetch_generic(url: str, domain: str) -> Dict[str, Any]:
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    # Handle PDF content regardless of Content-Type header
    if resp.content[:4] == b"%PDF":
        try:
            import io, PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(resp.content))
            text = ""
            for i in range(min(10, len(reader.pages))):
                text += reader.pages[i].extract_text() or ""
                if len(text) > 5000:
                    break
            if not text.strip():
                raise ValueError("empty extraction")
        except Exception as e:
            raise RuntimeError(f"PDF parse failed for {url}: {e}")
    else:
        text = resp.text

    return {
        "url": url,
        "title": f"{domain.title()} domain source: {url[:50]}",
        "body_text": text[:5000],
        "doc_type": "report",
        "authored_date": "2024-01-01",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE INSERTION
# ─────────────────────────────────────────────────────────────────────────────

def _insert_documents(domain: str, docs: List[Dict[str, Any]]) -> None:
    """
    Inserts fetched documents into the dossier.db documents table.
    Uses INSERT OR REPLACE so re-running the build is safe.
    """
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for i, doc in enumerate(docs):
        if not doc:
            continue
        doc_id = f"{domain.upper()}-DOC-{i+1:03d}"
        c.execute(
            """
            INSERT OR REPLACE INTO documents
            (doc_id, domain, doc_type, title, authored_date,
             received_date, version, source_url, body_text,
             key_claims, quantitative_fields, available_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                domain,
                doc.get("doc_type", "report"),
                doc.get("title", f"{domain} document {i+1}"),
                doc.get("authored_date", "2024-01-01"),
                doc.get("authored_date", "2024-01-01"),   # received same as authored
                "v1.0",
                doc.get("url", ""),
                doc.get("body_text", "")[:10000],         # cap body at 10k chars
                json.dumps([]),                           # key_claims populated below
                json.dumps({}),                           # quantitative_fields
                doc.get("authored_date", "2024-01-01"),
            ),
        )

    conn.commit()
    conn.close()


def _insert_distortions(distortions: List[Dict[str, Any]]) -> None:
    """
    Inserts generated distortion templates into dossier.db.
    Uses INSERT OR REPLACE so re-running is safe.
    """
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for d in distortions:
        c.execute(
            """
            INSERT OR REPLACE INTO distortions
            (distortion_id, source_doc_id, domain, distortion_type,
             original_claim, distorted_claim, detection_evidence, difficulty,
             requires_docs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d["distortion_id"],
                d["source_doc_id"],
                d["domain"],
                d["distortion_type"],
                d["original_claim"],
                d["distorted_claim"],
                d["detection_evidence"],
                d["difficulty"],
                json.dumps(d.get("requires_docs", [])),
            ),
        )

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  DISTORTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_distortions(
    domain: str,
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Calls Groq once to generate distortion templates from the fetched documents.
    Returns a list of distortion dicts ready for DB insertion.

    Distortion types generated:
      difficulty 1 — word_swap (direct word substitution)
      difficulty 2 — qualifier_strip (remove hedging language)
      difficulty 3 — attribution_error (misattribute source or conclusion)
      difficulty 4 — threshold_substitution (change numeric thresholds)

    Each distortion has:
      original_claim   — exact text from source document
      distorted_claim  — the modified version the questioner will present
      detection_evidence — what the witness can cite to catch it
    """
    if not docs:
        return _fallback_distortions(domain)

    # Concatenate source text for the prompt
    source_text = "\n\n---\n\n".join(
        f"[{d.get('title', '')}]\n{d.get('body_text', '')[:800]}"
        for d in docs if d
    )[:_MAX_SOURCE_CHARS]

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are generating adversarial distortion templates for an AI training environment.

Domain: {domain}
Source documents:
{source_text}

Generate exactly {_DISTORTIONS_PER_DOC * max(len(docs), 2)} distortion templates from the above text.
For each distortion, produce a JSON object with these fields:
  - original_claim: an exact or near-exact quote from the source text (1-2 sentences)
  - distorted_claim: a subtly modified version that changes the meaning
  - distortion_type: one of [word_swap, qualifier_strip, attribution_error, threshold_substitution]
  - difficulty: integer 1-4 (1=direct word change, 4=subtle threshold or attribution change)
  - detection_evidence: what text from the source proves the distortion is wrong

Return ONLY a valid JSON array. No preamble, no markdown, no explanation.
Example entry:
{{"original_claim": "The risk was assessed as moderate-to-elevated.", "distorted_claim": "The risk was assessed as low.", "distortion_type": "word_swap", "difficulty": 1, "detection_evidence": "Source states moderate-to-elevated, not low."}}"""

    try:
        resp = client.chat.completions.create(
            model=WITNESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2000,
        )
        content = resp.choices[0].message.content
        if not content or not content.strip():
            # Surface what Groq actually returned for debugging
            print(f"        Debug: Groq returned empty content. "
                  f"finish_reason={resp.choices[0].finish_reason}")
            raise ValueError("Empty response from LLM")

        # Strip all markdown and whitespace aggressively
        content = content.strip()
        # Remove ```json ... ``` or ``` ... ``` fences
        import re
        content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        content = content.strip()

        # Find the JSON array — scan for first '[' if there's preamble
        bracket_start = content.find('[')
        bracket_end = content.rfind(']')
        if bracket_start != -1 and bracket_end != -1:
            content = content[bracket_start:bracket_end + 1]

        raw = json.loads(content)

    except Exception as e:
        print(f"        Warning: LLM distortion generation failed ({e}). Using fallback.")
        return _fallback_distortions(domain)

    # Attach doc references and IDs
    distortions = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        doc_idx = i % max(len(docs), 1)
        source_doc_id = f"{domain.upper()}-DOC-{doc_idx+1:03d}"
        distortions.append({
            "distortion_id":    f"{domain.upper()}-DIST-{i+1:04d}-{uuid.uuid4().hex[:6]}",
            "source_doc_id":    source_doc_id,
            "domain":           domain,
            "distortion_type":  item.get("distortion_type", "word_swap"),
            "original_claim":   item.get("original_claim", ""),
            "distorted_claim":  item.get("distorted_claim", ""),
            "detection_evidence": item.get("detection_evidence", ""),
            "difficulty":       int(item.get("difficulty", 1)),
            "requires_docs":    [source_doc_id],
        })

    return distortions


def _fallback_distortions(domain: str) -> List[Dict[str, Any]]:
    """
    Hardcoded fallback distortions used when LLM generation fails or
    no source documents were fetched. One per difficulty level.
    These are minimal but cover all four distortion types so the
    environment is runnable in development without Groq access.
    """
    base_id = domain.upper()
    return [
        {
            "distortion_id":    f"{base_id}-FALLBACK-0001",
            "source_doc_id":    f"{base_id}-DOC-001",
            "domain":           domain,
            "distortion_type":  "word_swap",
            "original_claim":   f"The {domain} assessment indicated moderate risk.",
            "distorted_claim":  f"The {domain} assessment indicated low risk.",
            "detection_evidence": "Original assessment stated moderate, not low.",
            "difficulty":       1,
            "requires_docs":    [f"{base_id}-DOC-001"],
        },
        {
            "distortion_id":    f"{base_id}-FALLBACK-0002",
            "source_doc_id":    f"{base_id}-DOC-001",
            "domain":           domain,
            "distortion_type":  "qualifier_strip",
            "original_claim":   f"Findings were preliminary and subject to further review.",
            "distorted_claim":  f"Findings were definitive.",
            "detection_evidence": "Original stated preliminary and subject to review.",
            "difficulty":       2,
            "requires_docs":    [f"{base_id}-DOC-001"],
        },
        {
            "distortion_id":    f"{base_id}-FALLBACK-0003",
            "source_doc_id":    f"{base_id}-DOC-001",
            "domain":           domain,
            "distortion_type":  "attribution_error",
            "original_claim":   f"The panel recommended further investigation.",
            "distorted_claim":  f"You recommended further investigation.",
            "detection_evidence": "The recommendation came from the panel, not the witness.",
            "difficulty":       3,
            "requires_docs":    [f"{base_id}-DOC-001"],
        },
        {
            "distortion_id":    f"{base_id}-FALLBACK-0004",
            "source_doc_id":    f"{base_id}-DOC-001",
            "domain":           domain,
            "distortion_type":  "threshold_substitution",
            "original_claim":   f"The threshold was set at 0.05 significance level.",
            "distorted_claim":  f"The threshold was set at 0.10 significance level.",
            "detection_evidence": "Source states 0.05, not 0.10.",
            "difficulty":       4,
            "requires_docs":    [f"{base_id}-DOC-001"],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _check_api_key() -> None:
    """
    Validates that GROQ_API_KEY is set before starting the build.
    Fails fast with a clear error — building silently without the key
    would produce empty personas and fallback-only distortions.
    """
    if not os.getenv("GROQ_API_KEY"):
        print(
            "ERROR: GROQ_API_KEY not set.\n"
            "Add it to your .env file:\n"
            "  GROQ_API_KEY=gsk_...\n"
            "Then re-run: python scripts/build_dossier.py"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()