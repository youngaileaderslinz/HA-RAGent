from abc import ABC, abstractmethod
import json
from typing import Any

class SerializableModel(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, any]:
        """Return a dictionary representation of the embeddable model."""
        raise NotImplementedError("to_dict method must be implemented in subclasses.")

    @abstractmethod
    def from_dict(cls, data: dict[str, any]) -> "SerializableModel":
        """Create an instance of the database model from a dictionary representation."""
        raise NotImplementedError("from_dict method must be implemented in subclasses.")

    def from_json_str(cls, json_str: str) -> "SerializableModel":
        """Create an instance of the database model from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_json_str(self) -> str:
        """Serialize the model to a JSON string."""
        return json.dumps(self.to_dict())

    def to_prompt_dict(self, fields: dict[str, str]) -> str:
        """Return compact JSON with mapped field names and no empty values or nulls."""
        if not isinstance(fields, dict):
            raise TypeError("fields must be a dictionary mapping class field names to prompt names")
        candidate = self._exclude_null_values(self._prompt_data())
        return json.dumps({
            prompt_name: candidate[field] for field, prompt_name in fields.items()
            if field in candidate and candidate[field] not in (None, "", [], {})
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _prompt_data(self) -> dict[str, Any]:
        """Use class fields unless a model supplies normalized prompt data."""
        return vars(self)

    @classmethod
    def _exclude_null_values(cls, value: Any) -> Any:
        """Remove nulls from nested dictionaries and lists without mutating them."""
        if isinstance(value, dict):
            return {key: cls._exclude_null_values(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [cls._exclude_null_values(item) for item in value if item is not None]
        return value
