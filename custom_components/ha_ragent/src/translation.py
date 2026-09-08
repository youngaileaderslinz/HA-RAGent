from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


class RAGentTranslations:
    _cache: dict[str, dict[str, Any]] = {}

    def __init__(self, language: str = "en") -> None:
        self.language = language.split("-")[0].lower()
        self._data = self._cache.get(self.language, {"Prompts": {}, "Error messages": {}, "Tools": {}})

    @classmethod
    async def async_create(cls, hass: Any, language: str = "en") -> "RAGentTranslations":
        """Create a translation service without performing file I/O on the event loop."""
        normalized = language.split("-")[0].lower()
        if normalized not in cls._cache:
            cls._cache[normalized] = await hass.async_add_executor_job(cls._load, normalized)
        return cls(normalized)

    @staticmethod
    def _load(language: str) -> dict[str, Any]:
        if language in RAGentTranslations._cache:
            return RAGentTranslations._cache[language]
        package = files("custom_components.ha_ragent").joinpath(
            "translations", f"haragent_{language}.json"
        )
        try:
            data = json.loads(package.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if language != "en":
                return RAGentTranslations._load("en")
            data = {"Prompts": {}, "Error messages": {}, "Tools": {}}
        if not isinstance(data, dict):
            data = {"Prompts": {}, "Error messages": {}, "Tools": {}}
        RAGentTranslations._cache[language] = data
        return data

    def _section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def prompt(self, key: str) -> str:
        return str(self._section("Prompts")[key])

    def error(self, key: str, **values: Any) -> str:
        text = str(self._section("Error messages")[key])
        return text.format(**values) if values else text

    def tool(self, key: str) -> str:
        return str(self._section("Tools")[key])

    def has_tool(self, key: str) -> bool:
        """Return whether the integration owns a translation for a tool."""
        return key in self._section("Tools")

    def get(self, section: str, key: str, default: str = "") -> str:
        """Return a translated value by section name."""
        return str(self._section(section).get(key, default))
