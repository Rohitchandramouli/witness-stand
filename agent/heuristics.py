"""
Cross-episode heuristic library.
Logs successful response strategies so future episodes can draw on them.
Pattern: (distortion_type, context_signals) -> response_strategy
"""
import json
from pathlib import Path
from typing import Dict, List

HEURISTICS_PATH = Path("data/heuristics.json")


class HeuristicLibrary:
    def __init__(self):
        self._library: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if HEURISTICS_PATH.exists():
            with open(HEURISTICS_PATH) as f:
                self._library = json.load(f)

    def log_success(self, distortion_type: str, strategy_note: str) -> None:
        """Records a successful strategy for a given distortion type."""
        if distortion_type not in self._library:
            self._library[distortion_type] = []
        self._library[distortion_type].append(strategy_note)
        HEURISTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HEURISTICS_PATH, "w") as f:
            json.dump(self._library, f, indent=2)

    def get_hints(self, distortion_type: str) -> List[str]:
        """Returns past successful strategies for a distortion type."""
        return self._library.get(distortion_type, [])
