from abc import ABC, abstractmethod

class EmbeddableModel(ABC):
    @staticmethod
    def append_if_exists(parts_list: list[str], format_str: str, value: str | None) -> None:
        """Append a value to a list if it exists and is not empty."""
        if value:
            value_str = str(value) if not isinstance(value, list) else ", ".join(str(v) for v in value)
            parts_list.append(f"{format_str.format(value_str)}")

    @abstractmethod
    def to_embedding_text(self) -> str:
        """Return the semantic text sent to the embedding model."""
        raise NotImplementedError("to_embedding_text method must be implemented in subclasses.")
