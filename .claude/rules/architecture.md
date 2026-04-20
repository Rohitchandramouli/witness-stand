# Architecture rules

## Transcript is top-level
transcript/ is the shared spine read by 4 systems simultaneously.
Never move store.py or types.py inside another module.

## environment.py is a thin orchestrator
It delegates — never simulates. Max ~150 lines.
If logic grows beyond "call X, return Y", move it to the relevant module.

## Witness speaks from persona, not lookup
The witness answers from its system prompt (internalized persona).
Tool calls (search_record, retrieve_document, flag_inconsistency) are
for citation and verification only — never for answering questions.

## dossier/ vs tasks/
dossier/ constructs the persona from real documents.
tasks/ configures the episode (turns, lag, questioner panel).
These are separate. Never merge them.
