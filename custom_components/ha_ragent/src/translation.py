from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


class RAGentTranslations:
    _cache: dict[str, dict[str, Any]] = {}
    _supported_languages: tuple[str, ...] = ()

    def __init__(self, language: str = "en") -> None:
            self.language = language.split("-")[0].lower()
            self._data = self._cache.get(self.language, {"Prompts": {}, "Error messages": {}, "Tools": {}})

    @classmethod
    def supported_languages(cls) -> list[str]:
        """Return the asynchronously discovered supported language codes."""
        return list(cls._supported_languages)

    @staticmethod
    def _discover_supported_languages() -> tuple[str, ...]:
        """Find language pairs without blocking Home Assistant's event loop."""
        try:
            directory = files("custom_components.ha_ragent").joinpath("translations")
            names = {
                resource.name
                for resource in directory.iterdir()
                if resource.is_file() and resource.name.endswith(".json")
            }
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            return ()
        
        return tuple(sorted(
            name[len("haragent_"):-len(".json")]
            for name in names
            if (
                name.startswith("haragent_")
                and len(name) > len("haragent_.json")
                and f"{name[len('haragent_'):-len('.json')]}.json" in names
            )
        ))
    
    @classmethod
    async def async_create(cls, hass: Any, language: str = "en") -> "RAGentTranslations":
        """Create a translation service without performing file I/O on the event loop."""
        normalized = language.split("-")[0].lower()
        if not cls._supported_languages:
            cls._supported_languages = await hass.async_add_executor_job(cls._discover_supported_languages,)
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

    @classmethod
    def default(cls, language: str = "en") -> "RAGentTranslations":
        """Return a synchronously available translation service for pure models."""
        normalized = language.split("-")[0].lower()
        if normalized not in cls._cache:
            cls._load(normalized)
        return cls(normalized)

    def _get_section(self, name: str) -> dict[str, Any]:
        """Return a section of the translation data or an empty dict if not found."""
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def prompt(self, key: str) -> str:
        """Return a translated prompt by key."""
        return str(self._get_section("Prompts")[key])

    def error(self, key: str, **values: Any) -> str:
        """Return a translated error message by key, formatted with values."""
        text = str(self._get_section("Error messages")[key])
        return text.format(**values) if values else text

    def tool(self, key: str) -> str:
        """Return a translated tool name by key."""
        return str(self._get_section("Tools")[key])

    def embedding(self, key: str, **values: Any) -> str:
        """Render a translated embedding-text format."""
        text = str(self._get_section("Embedding")[key])
        return text.format(**values)

    def has_tool(self, key: str) -> bool:
        """Return whether the integration owns a translation for a tool."""
        return key in self._get_section("Tools")

    def get(self, section: str, key: str, default: str = "") -> str:
        """Return a translated value by section name."""
        return str(self._get_section(section).get(key, default))
