# Code style rules

## Types
- Use dataclasses for all internal models (not Pydantic)
- Use Pydantic only for OpenEnv State/Observation/Action wrappers
- Always add type hints to function signatures

## Comments
- Minimal and purposeful only
- No noise comments — if the name explains it, no comment needed
- No print statements in core modules (transcript/, grader/, agent/, questioners/)

## Imports
- No circular imports — follow build dependency order strictly
- models.py and constants.py import nothing from this project

## Functions
- Keep functions under 40 lines
- One clear responsibility per function
