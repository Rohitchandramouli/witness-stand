"""Base contract and shared helpers for all Witness Stand domain dossiers."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

from dossier.dossier_db import get_distortions_for_domain, search_record
from models import PersonaConfig


class DossierBase(ABC):
    domain: ClassVar[str] = ""
    source_urls: ClassVar[List[str]] = []

    @abstractmethod
    def get_persona_config(self) -> PersonaConfig:
        raise NotImplementedError

    def get_distortion_templates(self) -> List[Dict[str, Any]]:
        try:
            return get_distortions_for_domain(self.domain, [1, 2, 3, 4])
        except Exception as exc:
            print(f"Warning: could not load distortions for '{self.domain}': {exc}")
            return []

    @abstractmethod
    def get_false_authority_pool(self) -> List[str]:
        raise NotImplementedError

    def get_genuine_evidence(self) -> List[Dict[str, Any]]:
        try:
            return search_record("", domain=self.domain)
        except Exception as exc:
            print(f"Warning: could not load evidence for '{self.domain}': {exc}")
            return []