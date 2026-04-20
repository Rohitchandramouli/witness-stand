# /project:demo

Prints the before/after transcript contrast for the pitch demo.
Shows baseline agent capitulating vs trained agent catching distortion.

## Command
`ash
python scripts/demo.py
`
"@

New-File "C:\Users\rohit\OneDrive\Desktop\Meta_HF_Competition\witness_stand\.claude\commands\validate.md" @"
# /project:validate

Runs OpenEnv spec compliance check across all 4 tasks.

## Command
`ash
python scripts/validate.py
`

## Expected output
- PASS / FAIL per task
- Any spec violations listed with line references
