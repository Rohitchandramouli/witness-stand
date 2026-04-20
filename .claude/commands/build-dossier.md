# /project:build-dossier

Fetches real public documents, runs LLM extraction pass,
generates persona system prompts and distortions table.
Run this once before any training. Output goes to data/ (gitignored).

## Command
`ash
python scripts/build_dossier.py
`

## Expected output
- data/dossier.db (SQLite evidence archive)
- data/personas/ (one .json per domain)
- data/distortions/ (one .json per domain)

## Notes
- Requires internet access for fetching real documents
- Requires GROQ_API_KEY for the extraction LLM pass
- Takes 5-15 minutes first run
- Dockerfile runs this automatically at HF Space container startup
