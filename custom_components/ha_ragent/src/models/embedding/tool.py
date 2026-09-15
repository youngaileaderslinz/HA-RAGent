from __future__ import annotations

from typing import Any, Dict
import json
from dataclasses import dataclass
from copy import deepcopy

from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.embedding.tool_metadata import (
    ToolMetadata,
    split_canonical_name,
)
from custom_components.ha_ragent.src.translation import RAGentTranslations

@dataclass
class LlmTool(SerializableModel, EmbeddableModel):
    name: str
    description: str
    metadata: ToolMetadata = None
    parameters: Dict[str, Any] = None

    @property
    def canonical_name_parts(self) -> tuple[str, ...]:
        """Get the canonical parts of the tool's name."""
        return self.split_canonical_name(self.name)

    @property
    def canonical_action(self) -> str:
        """Return an explicitly declared capability, never inferred from prose."""
        return self.metadata.canonical_action if self.metadata else ""

    @staticmethod
    def _schema_values(schema: object) -> set[str]:
        values: set[str] = set()
        if isinstance(schema, dict):
            for name, value in schema.items():
                if name == "const" and isinstance(value, (str, int, float)):
                    values.add(str(value).casefold())
                elif name == "enum" and isinstance(value, list):
                    values.update(str(item).casefold() for item in value)
                else:
                    values.update(LlmTool._schema_values(value))
        elif isinstance(schema, list):
            for value in schema:
                values.update(LlmTool._schema_values(value))
        return values

    @classmethod
    def _schema_field_values(cls, schema: object, field: str) -> set[str]:
        """Collect constrained values for a field throughout a nested schema."""
        values: set[str] = set()
        if isinstance(schema, dict):
            for name, value in schema.items():
                if name == field:
                    values.update(cls._schema_values(value))
                values.update(cls._schema_field_values(value, field))
        elif isinstance(schema, list):
            for value in schema:
                values.update(cls._schema_field_values(value, field))
        return values

    @staticmethod
    def _schema_search_parts(schema: object, path: str = "", depth: int = 0) -> tuple[str, ...]:
        """Return bounded names, descriptions, required fields, types, and enums."""
        if depth > 4 or not isinstance(schema, dict):
            return ()
        parts: list[str] = []
        description = schema.get("description")
        schema_type = schema.get("type")
        required = schema.get("required")
        enum = schema.get("enum")
        if path:
            parts.append(f"parameter {path}")
        if isinstance(description, str) and description:
            parts.append(description)
        if schema_type:
            parts.append(f"type {schema_type}")
        if isinstance(required, list) and required:
            parts.append("required " + " ".join(str(name) for name in required))
        if isinstance(enum, list) and enum:
            parts.append("choices " + " ".join(str(value) for value in enum))
        if "const" in schema:
            parts.append(f"constant {schema['const']}")
        for keyword in ("anyOf", "oneOf", "allOf"):
            for variant in schema.get(keyword, []):
                parts.extend(LlmTool._schema_search_parts(variant, path, depth + 1))
        for name, value in (schema.get("properties") or {}).items():
            child_path = f"{path}.{name}" if path else str(name)
            parts.extend(LlmTool._schema_search_parts(value, child_path, depth + 1))
        items = schema.get("items")
        if isinstance(items, dict):
            parts.extend(LlmTool._schema_search_parts(items, f"{path} item".strip(), depth + 1))
        return tuple(parts[:80])

    @property
    def canonical_schema_parts(self) -> tuple[str, ...]:
        """Return searchable live-schema metadata."""
        return self._schema_features[1]

    @property
    def _schema_features(self) -> tuple:
        """Reuse derived schema fields; detect nested edits as well as replacement."""
        parameters = self.parameters or {}
        cached = getattr(self, "_cached_schema_features", None)
        if cached is None or cached[0] != parameters:
            snapshot = deepcopy(parameters)
            cached = (
                snapshot,
                self._schema_search_parts(snapshot),
                frozenset(self._schema_field_values(snapshot, "domain")),
                frozenset(self._schema_field_values(snapshot, "device_class")),
            )
            self._cached_schema_features = cached
        return cached

    @property
    def schema_domains(self) -> frozenset[str]:
        return self._schema_features[2]

    @property
    def schema_device_classes(self) -> frozenset[str]:
        return self._schema_features[3]

    @property
    def canonical_supported_domains(self) -> tuple[str, ...]:
        """Return explicit target domains declared by the tool schema."""
        domains = set(self.schema_domains)
        if self.metadata:
            metadata_domains = (
                self.metadata.get("supported_domains", ())
                if isinstance(self.metadata, dict)
                else self.metadata.supported_domains
            )
            domains.update(metadata_domains or ())
        return tuple(sorted(domains))

    @property
    def canonical_search_parts(self) -> tuple[str, ...]:
        """Return searchable identity, capability, domain, and schema metadata."""
        canonical_name = " ".join(self.canonical_name_parts)
        return tuple(
            value
            for value in (
                self.name,
                canonical_name,
                self.canonical_action,
                *self.canonical_supported_domains,
                self.description,
                *self.canonical_schema_parts,
            )
            if value
        )

    @property
    def canonical_action_document(self) -> str:
        """Return compact language-independent action metadata for lexical retrieval."""
        parts: list[object] = [
            self.canonical_action or " ".join(self.canonical_name_parts),
            *self.canonical_supported_domains,
        ]
        if self.metadata:
            expected_states = (
                self.metadata.get("expected_states", ())
                if isinstance(self.metadata, dict)
                else self.metadata.expected_states
            )
            parts.extend(expected_states or ())
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def split_canonical_name(name: str) -> tuple[str, ...]:
        """Split on underscores and camel-case transitions."""
        return split_canonical_name(name)
    
    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the tool."""
        metadata = self.metadata.to_dict() if self.metadata else None
        if metadata is not None:
            # Existing indexes can contain metadata written before schema
            # domains were extracted. Always serialize the live schema-derived
            # value so re-exposure does not preserve stale empty metadata.
            metadata["supported_domains"] = list(self.canonical_supported_domains)
        return {
            "name": self.name,
            "description": self.description,
            "metadata": json.dumps(metadata) if metadata is not None else None,
            "parameters": json.dumps(self.parameters)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LlmTool':
        """Create an LlmTool instance from a dictionary."""
        metadata = json.loads(data["metadata"]) if data.get("metadata") else None
        parameters = json.loads(data["parameters"]) if data.get("parameters") else None
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            metadata=ToolMetadata.from_dict(metadata) if metadata else None,
            parameters=parameters,
        )

    def to_embedding_text(self, translations: RAGentTranslations | None = None) -> str:
        """Return a compact action concept for multilingual semantic search."""
        translations = self._translations(translations)
        parts = []

        self.append_if_exists(parts, translations.embedding("tool_action", value="{}"), self.canonical_action)
        self.append_if_exists(parts, translations.embedding("tool_domains", value="{}"), list(self.canonical_supported_domains))

        if self.metadata:
            expected_states = self.metadata.get("expected_states", ()) if isinstance(self.metadata, dict) else self.metadata.expected_states
            self.append_if_exists(parts, translations.embedding("tool_expected_states", value="{}"), list(expected_states or ()))

        self.append_if_exists(parts, translations.embedding("tool_description", value="{}"), self.description)

        return " ".join(parts)

    def to_tool_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool suitable for use in an LLM context."""
        tool_def = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
            }
        }

        if self.parameters:
            tool_def["function"]["parameters"] = self.parameters

        return tool_def
