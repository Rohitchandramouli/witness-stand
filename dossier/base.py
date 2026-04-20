"""Abstract base for all domain dossiers."""
from abc import ABC, abstractmethod
from typing import List, Dict
from models import PersonaConfig


class DossierBase(ABC):
    domain: str = ""
    source_urls: List[str] = []

    @abstractmethod
    def get_persona_config(self) -> PersonaConfig:
        """Returns the persona identity and knowledge config for this domain."""
        ...

    @abstractmethod
    def get_distortion_templates(self) -> List[Dict]:
        """Returns distortion templates generated from real document content."""
        ...

    @abstractmethod
    def get_false_authority_pool(self) -> List[str]:
        """Returns list of fabricated expert names for the Authority Invoker."""
        ...

    @abstractmethod
    def get_genuine_evidence(self) -> List[Dict]:
        """Returns genuine new evidence items that should cause real updates."""
        ...
