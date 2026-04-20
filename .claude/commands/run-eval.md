# /project:run-eval

Runs the full 4-task benchmark with ELO tracking.

## Command
`ash
python scripts/run_eval.py
`

## Expected output
- Per-task scores for basic / intermediate / advanced / expert
- ELO rating for Witness and Questioner
- Final episode score in [0.0, 1.0]

## If it fails
- Check that data/ exists (run /project:build-dossier first)
- Check GROQ_API_KEY is set in environment
- Check openenv spec passes (/project:validate)
