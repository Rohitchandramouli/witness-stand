"""Episodic witness memory."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from models import Speaker, Turn


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "that", "this",
    "there", "their", "with", "from", "into", "about", "your", "you",
    "was", "were", "is", "are", "be", "been", "being", "for", "to", "of",
    "in", "on", "at", "by", "as", "it", "my", "i", "we", "our",
}


class EpisodicMemory:
    """Stores and retrieves witness statements within one episode."""

    def __init__(self, max_turns: int = 40) -> None:
        self._turns: List[Turn] = []
        self._max_turns = max(1, max_turns)

    def store(self, turn: Turn) -> None:
        """Stores only witness turns."""
        if turn.speaker != Speaker.WITNESS:
            return

        clean_text = " ".join(str(turn.text).split())
        turn.text = clean_text
        self._turns.append(turn)

        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def get_by_rank(self, rank: int) -> Optional[Turn]:
        """Rank 0 = most recent, rank 1 = second most recent."""
        if rank < 0 or rank >= len(self._turns):
            return None
        return self._turns[-(rank + 1)]

    def get_recent(self, n: int = 3) -> List[Turn]:
        """Returns the n most recent witness turns in chronological order."""
        if n <= 0:
            return []
        return list(self._turns[-n:])

    def get_visible(self, data_lag_turns: int = 0, recent_n: Optional[int] = None) -> List[Turn]:
        """Returns visible memory after applying data lag and optional recent window."""
        if data_lag_turns > 0:
            visible = self._turns[:-data_lag_turns] if len(self._turns) > data_lag_turns else []
        else:
            visible = self._turns

        if recent_n is not None and recent_n > 0:
            visible = visible[-recent_n:]

        return list(visible)

    def get_all(self) -> List[Turn]:
        """Returns all witness turns in chronological order."""
        return list(self._turns)

    def get_by_turn_no(self, turn_no: int) -> Optional[Turn]:
        """Returns a witness statement by original turn number."""
        for turn in reversed(self._turns):
            if turn.turn_no == turn_no:
                return turn
        return None

    def find_claim(self, keyword: str) -> Optional[Turn]:
        """Returns the most recent turn matching a keyword or phrase."""
        matches = self.find_claims(keyword, top_k=1)
        return matches[0] if matches else None

    def find_claims(self, query: str, top_k: int = 3) -> List[Turn]:
        """Returns best matching witness turns using deterministic token overlap."""
        if not query or not query.strip():
            return []

        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        scored: List[Tuple[float, int, Turn]] = []

        for idx, turn in enumerate(self._turns):
            turn_tokens = self._tokens(turn.text)
            if not turn_tokens:
                continue

            overlap = query_tokens & turn_tokens
            if not overlap:
                continue

            score = len(overlap) / max(len(query_tokens), 1)
            recency_bonus = idx / max(len(self._turns), 1) * 0.05
            scored.append((score + recency_bonus, turn.turn_no, turn))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [turn for _, _, turn in scored[:top_k]]

    def summary(self, max_chars_per_turn: int = 160, recent_n: Optional[int] = None) -> str:
        """Returns a compact memory summary for prompts/debugging."""
        turns = self._turns[-recent_n:] if recent_n else self._turns

        if not turns:
            return "(No witness statements stored.)"

        lines = []
        for turn in turns:
            text = self._compact(turn.text, max_chars_per_turn)
            lines.append(f"[Turn {turn.turn_no}] {text}")

        return "\\n".join(lines)

    def reset(self) -> None:
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"EpisodicMemory(turns={len(self._turns)}, max={self._max_turns})"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        raw_tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\\-']+", text.lower())
        return {
            token
            for token in raw_tokens
            if len(token) > 2 and token not in _STOPWORDS
        }

    @staticmethod
    def _compact(text: str, max_chars: int) -> str:
        clean = " ".join(str(text).split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."
