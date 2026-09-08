import re
from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.const import CANONICAL_NAME_SPLIT_PATTERN
from custom_components.ha_ragent.src.models.retrieval.lexical_index import normalize
from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel


def split_canonical_name(name: str) -> tuple[str, ...]:
    """Split a canonical name on underscores and camel-case transitions."""
    return tuple(
        part.casefold()
        for part in re.split(CANONICAL_NAME_SPLIT_PATTERN, str(name or ""))
        if part
    )


def normalize_canonical_text(text: object) -> str:
    """Normalize free text for canonical capability matching."""
    return normalize(re.sub(CANONICAL_NAME_SPLIT_PATTERN, " ", str(text or "")))


@dataclass
class ToolMetadata(SerializableModel):
    family: str = None
    is_domain_aware: bool = False
    is_area_aware: bool = False
    is_device_class_aware: bool = False

    canonical_action: str = ""
    supported_domains: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the tool metadata."""
        return {
            "family": self.family,
            "canonical_action": self.canonical_action,
            "supported_domains": list(self.supported_domains),
            "is_domain_aware": self.is_domain_aware,
            "is_area_aware": self.is_area_aware,
            "is_device_class_aware": self.is_device_class_aware,  
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> 'ToolMetadata':
        """Create a ToolMetadata instance from a dictionary."""
        return ToolMetadata(
            family=data.get("family", None),
            canonical_action=data.get("canonical_action", ""),
            supported_domains=tuple(data.get("supported_domains") or ()),
            is_domain_aware=data.get("is_domain_aware", False),
            is_area_aware=data.get("is_area_aware", False),
            is_device_class_aware=data.get("is_device_class_aware", False)
        )
