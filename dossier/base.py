"""Abstract base for all domain dossiers."""
from abc import ABC, abstractmethod
from typing import List, Dict
from models import PersonaConfig


class DossierBase(ABC):
    domain: str = ""
    source_urls: List[str] = []

    @abstractmethod
    def get_persona_config(self) -> PersonaConfig:
        ...

    @abstractmethod
    def get_distortion_templates(self) -> List[Dict]:
        ...

    @abstractmethod
    def get_false_authority_pool(self) -> List[str]:
        ...

    @abstractmethod
    def get_genuine_evidence(self) -> List[Dict]:
        ...