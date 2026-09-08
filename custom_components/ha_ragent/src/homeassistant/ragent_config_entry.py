from homeassistant.config_entries import ConfigEntry

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.translation import RAGentTranslations

class RAGentConfigEntry(ConfigEntry):
    """RAGent Config Entry"""
    vector_db_backend: ABaseDbBackend
    embedder_backend: ABaseEmbedder
    llm_backend: ALlmBaseBackend
    translations: RAGentTranslations
