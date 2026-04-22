"""
persona_builder.py — one-time offline LLM extraction pass.
Fetches real documents from source_urls, extracts key claims,
synthesises identity + domain knowledge + position memory into system prompt.
Output goes to data/personas/ (gitignored, rebuilt at container startup).
"""
import json
import os
import requests
from pathlib import Path
from typing import Type, Dict, Any
from dotenv import load_dotenv
from groq import Groq

from dossier.base import DossierBase
from constants import PERSONAS_DIR, WITNESS_MODEL

load_dotenv()


def build_persona(dossier_class: Type[DossierBase]) -> Dict[str, Any]:
    """
    Builds and saves a complete persona system prompt for a given dossier domain.
    Fetches real documents from source_urls, runs LLM extraction to synthesise
    identity + domain knowledge + position memory into a system prompt.
    Output saved to data/personas/<domain>.json (gitignored).
    Runs once offline before any training episode begins.
    """
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)

    dossier = dossier_class()
    config = dossier.get_persona_config()
    output_path = PERSONAS_DIR / f"{dossier.domain}.json"

    raw_documents = _fetch_documents(dossier)
    system_prompt = _synthesise_persona(config, raw_documents)

    config.system_prompt = system_prompt
    output = {
        "domain": dossier.domain,
        "persona": {
            "domain": config.domain,
            "name": config.name,
            "title": config.title,
            "institution": config.institution,
            "years_experience": config.years_experience,
            "specialisation": config.specialisation,
            "professional_philosophy": config.professional_philosophy,
            "system_prompt": config.system_prompt,
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Built persona: {dossier.domain} -> {output_path}")
    return output


def load_persona(domain: str) -> Dict[str, Any]:
    """
    Loads a pre-built persona from data/personas/<domain>.json.
    Called by task files at episode reset time.
    Raises FileNotFoundError with a clear message if build_dossier.py
    has not been run yet.
    """
    path = PERSONAS_DIR / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Persona not found for domain '{domain}' at {path}. "
            f"Run: python scripts/build_dossier.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fetch_documents(dossier: DossierBase) -> str:
    """
    Fetches a sample of real documents from the dossier's source_urls.
    Returns a single concatenated text block for the LLM extraction pass.
    Handles each source type with appropriate fetching logic.
    Failures are caught and logged — a missing source reduces persona richness
    but does not abort the build.
    """
    collected = []

    for url in dossier.source_urls:
        try:
            text = _fetch_single_source(url, dossier.domain)
            if text:
                collected.append(f"[Source: {url}]\n{text}")
        except Exception as e:
            print(f"  Warning: failed to fetch {url}: {e}")

    if not collected:
        print(f"  Warning: no documents fetched for {dossier.domain}. "
              f"Persona will be built from config fields only.")

    return "\n\n---\n\n".join(collected)


def _fetch_single_source(url: str, domain: str) -> str:
    """
    Routes each URL to the appropriate fetcher based on the source type.
    Returns plain text suitable for the LLM extraction pass.
    """
    # HuggingFace Hub API — fetch model cards with bias evaluation sections
    if "huggingface.co/api/models" in url:
        return _fetch_hf_model_cards()

    # ClinicalTrials.gov v2 API — fetch completed Indian trials
    elif "clinicaltrials.gov/api/v2" in url:
        return _fetch_clinical_trials()

    # PubMed E-utilities — fetch abstracts for domain-relevant papers
    elif "eutils.ncbi.nlm.nih.gov" in url:
        return _fetch_pubmed_abstracts(domain)

    # NTSB accident reports page — fetch recent investigation report index
    elif "ntsb.gov" in url:
        return _fetch_ntsb_reports()

    # NTSB downloadable aviation data zip
    elif "data.ntsb.gov/avdata" in url:
        return _fetch_ntsb_avdata()

    # Generic PDF or HTML page — attempt direct fetch and text extraction
    else:
        return _fetch_generic(url)


def _fetch_hf_model_cards() -> str:
    """Fetches 10 high-download models with detailed bias evaluation sections."""
    resp = requests.get(
        "https://huggingface.co/api/models",
        params={
            "sort": "downloads",
            "direction": "-1",
            "limit": 10,
            "full": "true",
        },
        timeout=15,
    )
    resp.raise_for_status()
    models = resp.json()

    texts = []
    for model in models:
        model_id = model.get("id", "")
        card_data = model.get("cardData", {})
        tags = model.get("tags", [])
        downloads = model.get("downloads", 0)

        snippet = (
            f"Model: {model_id}\n"
            f"Downloads: {downloads}\n"
            f"Tags: {', '.join(tags[:10])}\n"
            f"License: {card_data.get('license', 'unspecified')}\n"
            f"Language: {card_data.get('language', 'unspecified')}\n"
            f"Datasets: {card_data.get('datasets', [])}\n"
        )
        texts.append(snippet)

    return "\n".join(texts)


def _fetch_clinical_trials() -> str:
    """Fetches 10 completed Indian clinical trials with results."""
    resp = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={
            "query.locn": "India",
            "filter.overallStatus": "COMPLETED",
            "fields": "NCTId,BriefTitle,Condition,InterventionName,"
                      "PrimaryOutcomeMeasure,Phase,EnrollmentCount,"
                      "StartDate,CompletionDate",
            "pageSize": 10,
            "format": "json",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    studies = data.get("studies", [])

    texts = []
    for study in studies:
        proto = study.get("protocolSection", {})
        id_module = proto.get("identificationModule", {})
        status_module = proto.get("statusModule", {})
        design_module = proto.get("designModule", {})

        snippet = (
            f"Trial ID: {id_module.get('nctId', 'N/A')}\n"
            f"Title: {id_module.get('briefTitle', 'N/A')}\n"
            f"Phase: {design_module.get('phases', ['N/A'])}\n"
            f"Status: {status_module.get('overallStatus', 'N/A')}\n"
            f"Start: {status_module.get('startDateStruct', {}).get('date', 'N/A')}\n"
            f"Completion: {status_module.get('completionDateStruct', {}).get('date', 'N/A')}\n"
        )
        texts.append(snippet)

    return "\n".join(texts)


def _fetch_pubmed_abstracts(domain: str) -> str:
    """Fetches 5 PubMed abstracts relevant to the domain."""
    domain_queries = {
        "medical": "clinical trial India adverse events pharmacology",
        "financial": "NBFC risk assessment India financial stability",
        "safety": "industrial safety incident investigation India",
        "technical": "AI fairness bias audit algorithmic accountability",
    }
    query = domain_queries.get(domain, "AI safety India")

    # Step 1: search for IDs
    search_resp = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": 5,
            "retmode": "json",
        },
        timeout=15,
    )
    search_resp.raise_for_status()
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return ""

    # Step 2: fetch abstracts for those IDs
    fetch_resp = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "text",
        },
        timeout=15,
    )
    fetch_resp.raise_for_status()
    return fetch_resp.text[:3000]


def _fetch_ntsb_reports() -> str:
    """Fetches the NTSB investigations page as plain text."""
    resp = requests.get(
        "https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx",
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    # Return raw HTML truncated — LLM extraction pass will pull structure from it
    return resp.text[:5000]


def _fetch_ntsb_avdata() -> str:
    """Returns metadata about the NTSB downloadable aviation data set."""
    return (
        "NTSB Aviation Accident Database (avdata): structured accident records "
        "from 1982 to present. Fields include: event_id, event_date, location, "
        "country, air_carrier, aircraft_damage, injury_severity, probable_cause, "
        "contributing_factors, weather_condition, flight_phase. "
        "Available at: https://data.ntsb.gov/avdata"
    )


def _fetch_generic(url: str) -> str:
    """Generic HTTP GET with a 5000-character truncation."""
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    return resp.text[:5000]


def _synthesise_persona(config, raw_documents: str) -> str:
    """
    Runs the one-time LLM extraction pass via Groq.
    Takes the PersonaConfig fields and raw fetched document text,
    returns a rich system prompt the witness LLM receives each episode.
    This is the only LLM call in the entire build pipeline.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are building the persona system prompt for an expert witness AI agent.

The witness is: {config.name}, {config.title} at {config.institution}.
Years of experience: {config.years_experience}
Specialisation: {', '.join(config.specialisation)}
Professional philosophy: {config.professional_philosophy}

Below are real documents from the witness's domain. Use them to enrich the persona
with specific domain knowledge, realistic professional language, and concrete
examples of the kinds of assessments, findings, and positions this expert would hold.

REAL DOMAIN DOCUMENTS:
{raw_documents[:6000]}

Write a first-person system prompt for this witness. The prompt must:
1. Establish the witness's identity, institution, and professional history
2. Describe their methodology and analytical frameworks in specific terms
3. Include 3-5 concrete examples of the kinds of assessments they have made,
   grounded in the domain documents above where possible
4. Establish their professional values — particularly how they handle pressure
   to revise positions without new evidence
5. Include a position memory section with 2-3 specific prior statements they
   have made on record, with exact language and caveats they would defend
6. Make clear that they speak from internalized expertise — they do NOT look
   things up to answer questions, they know their work

Write in second person ("You are..."). Be specific and domain-grounded.
Length: 600-900 words."""

    response = client.chat.completions.create(
        model=WITNESS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError(
            f"Groq returned empty content for persona synthesis. "
            f"Check your GROQ_API_KEY in .env and try again."
        )
    return content.strip()