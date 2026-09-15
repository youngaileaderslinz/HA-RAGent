from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from custom_components.ha_ragent.src.translation import RAGentTranslations

class EmbeddableModel(ABC):
    @staticmethod
    def _translations(translations: "RAGentTranslations | None") -> "RAGentTranslations":
        """Use the entry translation service, or English for direct callers."""
        if translations is not None:
            return translations

        from custom_components.ha_ragent.src.translation import RAGentTranslations

        return RAGentTranslations.default("en")

    @staticmethod
    def append_if_exists(parts_list: list[str], format_str: str, value: str | None) -> None:
        """Append a value to a list if it exists and is not empty."""
        if value:
            value_str = str(value) if not isinstance(value, list) else ", ".join(str(v) for v in value)
            parts_list.append(f"{format_str.format(value_str)}")

    @abstractmethod
    def to_embedding_text(self, translations: "RAGentTranslations | None" = None) -> str:
        """Return the semantic text sent to the embedding model."""
        raise NotImplementedError("to_embedding_text method must be implemented in subclasses.")
