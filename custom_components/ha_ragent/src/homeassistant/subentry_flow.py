import logging
import os
from typing import Any
import voluptuous as vol
from types import SimpleNamespace

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.config_entries import (
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult
)

from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend

from custom_components.ha_ragent.src.const import (
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_LLM_BACKEND_TYPE,
    CONF_EMBEDDING_MODEL,
    CONF_LLM_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_SELECTED_LANGUAGE,

    DEFAULT_SETTINGS,
    CONF_EXCLUDED_TOOLS,
    CONF_NUM_MEMORIES_TO_EXTRACT,
    CONF_MAX_MEMORY_ENTRIES
)

from custom_components.ha_ragent.src.utils import (
    try_parse_int, get_setting_value
)

from custom_components.ha_ragent.src.homeassistant.ui_schemas import (
    ui_schema_pick_models,
    ui_schema_config_options
)

from custom_components.ha_ragent.src.homeassistant.ragent import RAGent
from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor

_logger = logging.getLogger(__name__)

class RagentSubentryFlowHandler(ConfigSubentryFlow):
    def __init__(self) -> None:
        super().__init__()

        self.model_config: dict[str, Any] = {}
        self.download_task = None
        self.download_error = None

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    @property
    def _embedding_client(self) -> ABaseEmbedder:
        entry: RAGentConfigEntry = self._get_entry()
        return entry.embedder_backend

    @property
    def _llm_client(self) -> ALlmBaseBackend:
        entry: RAGentConfigEntry = self._get_entry()
        return entry.llm_backend

    async def async_step_pick_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        schema = vol.Schema({})
        errors = {}
        description_placeholders = {}

        embedding_models = await self._embedding_client.async_get_available_models()
        llm_models = await self._llm_client.async_get_available_models()
        _logger.debug("Available embedding models: %s", embedding_models)
        _logger.debug("Available LLM models: %s", llm_models)
        schema = ui_schema_pick_models(
            embedding_models,
            llm_models,
            embedding_model=self.model_config.get(CONF_EMBEDDING_MODEL),
            llm_model=self.model_config.get(CONF_LLM_MODEL),
        )

        if user_input and "result" not in user_input:
            self.model_config.update(user_input)
            return await self.async_step_model_parameters()

        return self.async_show_form(
            step_id="pick_model",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=False,
        )

    async def async_step_model_parameters(
        self, user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        errors = {}
        description_placeholders = {}
        entry = self._get_entry()

        if CONF_PROMPT not in self.model_config:
            selected_language = self.model_config.get(
                CONF_SELECTED_LANGUAGE, get_setting_value(CONF_SELECTED_LANGUAGE, entry.options)
            )

            selected_default_options = {**DEFAULT_SETTINGS}
            selected_default_options[CONF_PROMPT] = RAGent.build_base_prompt_template(selected_language, get_setting_value(CONF_PROMPT, selected_default_options))
            self.model_config = {**selected_default_options, **self.model_config}

        excluded_tool_names = { name for name in self.model_config.get(CONF_EXCLUDED_TOOLS, []) if isinstance(name, str) }
        try:
            subentry = self._get_reconfigure_subentry() if not self._is_new else SimpleNamespace(data=self.model_config, title="new subentry")
            excluded_tool_names.update(await ToolExtractor(self.hass, entry).async_get_embeddable_tool_names(subentry))
        except Exception:
            _logger.exception("Failed to load extracted tools for exclusion selector")

        schema = ui_schema_config_options(
                self.hass,
                entry.options.get(CONF_SELECTED_LANGUAGE, "en"),
                self.model_config,
                entry.data[CONF_VECTOR_DB_BACKEND_TYPE],
                entry.data[CONF_EMBEDDING_BACKEND_TYPE],
                entry.data[CONF_LLM_BACKEND_TYPE],
                self._subentry_type,
                list(excluded_tool_names),
            )

        if user_input:
            for key in (
                CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
                CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
                CONF_MAX_TOOL_CALL_ITERATIONS,
                CONF_CONTEXT_LENGTH,
                CONF_MAX_TOKENS,
                CONF_NUM_MEMORIES_TO_EXTRACT,
                CONF_MAX_MEMORY_ENTRIES,
             ):
                if key in user_input:
                    user_input[key] = try_parse_int(user_input[key], user_input.get(key) or 0)
            
            if len(errors) == 0:
                try:
                    schema(user_input)
                    self.model_config.update(user_input)
                    
                    return await self.async_step_finish()
                except Exception:
                    _logger.exception("An unknown error has occurred!")
                    errors["base"] = "unknown"

        return self.async_show_form(
            step_id="model_parameters",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._is_new:
            return self.async_create_entry(
                title=f"{self.model_config.get(CONF_EMBEDDING_MODEL, "Embedder")} + {self.model_config.get(CONF_LLM_MODEL, "LLM")}",
                data=self.model_config,
            )
        else:
            return self.async_update_and_abort(
                self._get_entry(), self._get_reconfigure_subentry(), data=self.model_config
            )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        
        if not self.model_config:
            self.model_config = {}
        
        return await self.async_step_pick_model(user_input)
    
    async_step_init = async_step_user
        
    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None):
        if not self.model_config:
            self.model_config = dict(self._get_reconfigure_subentry().data)

        return await self.async_step_model_parameters(user_input)
