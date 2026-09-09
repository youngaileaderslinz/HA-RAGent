from __future__ import annotations

import logging
from typing import Any

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.config_entries import (
    ConfigEntriesFlowManager,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
    ConfigSubentryFlow
)

from .src.homeassistant.ragent_config_entry import RAGentConfigEntry

from .src.const import (
    BACKEND_VECTOR_DB_TYPE_FAISS,
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_VECTOR_DB_NAME,
    DOMAIN,
    CONF_LLM_BACKEND_TYPE,
    CONF_SELECTED_LANGUAGE,
    
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_LLM_HOST,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONFIG_FLOW_VERSION,
)

from .src.homeassistant.option_flow import RagentOptionsFlow
from .src.homeassistant.subentry_flow import RagentSubentryFlowHandler

from .src.homeassistant.ui_schemas import (
    ui_schema_backend_connections,
    ui_schema_pick_backends
)

from .src.utils import (
    is_valid_host,
    vector_db_to_class,
    embedding_backend_to_class,
    llm_backend_to_class
)

_logger = logging.getLogger(__name__)

class RagentConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = CONFIG_FLOW_VERSION

    def __init__(self) -> None:
        super().__init__()
        self.client_config: dict[str, Any] = {}
        self.flow_step: str = "init"

    @property
    def flow_manager(self) -> ConfigEntriesFlowManager:
        return self.hass.config_entries.flow
            
    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        match self.flow_step:
            case "init": return await self._init_flow_async()
            case "configure_backend": return await self._configure_backend_async(user_input)
            case "connect_to_backend": return await self._connect_to_backend_async(user_input)
            case _: return self.async_abort(reason="unknown_step") 
                
    async def _init_flow_async(self) -> ConfigFlowResult:
        self.flow_step = "configure_backend"
        return self.async_show_form(
            step_id="user",
            data_schema=ui_schema_pick_backends(),
            last_step=False
        )
            
    async def _configure_backend_async(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input:
            self.client_config.update(user_input)
            self.flow_step = "connect_to_backend"
            return self.async_show_form(
                step_id="user", 
                data_schema=ui_schema_backend_connections(
                    vector_db_backend_type=self.client_config[CONF_VECTOR_DB_BACKEND_TYPE],
                    embedding_backend_type=self.client_config[CONF_EMBEDDING_BACKEND_TYPE],
                    llm_backend_type=self.client_config[CONF_LLM_BACKEND_TYPE]),
                last_step=True
            )
        return self.async_show_form(
            step_id="user", 
            data_schema=ui_schema_pick_backends(
                ventor_db_backend_type=self.client_config.get(CONF_VECTOR_DB_BACKEND_TYPE),
                embedding_backend_type=self.client_config.get(CONF_EMBEDDING_BACKEND_TYPE),
                llm_backend_type=self.client_config.get(CONF_LLM_BACKEND_TYPE),
                selected_language=self.client_config.get(CONF_SELECTED_LANGUAGE)), 
            last_step=False)
        
    async def _connect_to_backend_async(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors = {}
        description_placeholders = {}
        
        if user_input:
            self.client_config.update(user_input)
            vector_db_hostname = user_input.get(CONF_VECTOR_DB_HOST)
            embedding_hostname = user_input.get(CONF_EMBEDDING_HOST)
            llm_hostname = user_input.get(CONF_LLM_HOST)
            
            vector_db_is_valid = self.client_config.get(CONF_VECTOR_DB_BACKEND_TYPE) == BACKEND_VECTOR_DB_TYPE_FAISS or is_valid_host(vector_db_hostname)
            embedding_is_valid = is_valid_host(embedding_hostname)
            llm_is_valid = is_valid_host(llm_hostname)

            if not vector_db_is_valid or not embedding_is_valid or not llm_is_valid:
                errors["base"] = "invalid_hostname"
                description_placeholders["exception"] = "The provided hostname could not be resolved to an IP address."
            else:
                connect_err = await vector_db_to_class(self.client_config.get(CONF_VECTOR_DB_BACKEND_TYPE)).async_validate_connection(self.hass, self.client_config)

                if not connect_err:
                    connect_err = await embedding_backend_to_class(self.client_config.get(CONF_EMBEDDING_BACKEND_TYPE)).async_validate_connection(self.hass, self.client_config)
                
                if not connect_err:
                    connect_err = await llm_backend_to_class(self.client_config.get(CONF_LLM_BACKEND_TYPE)).async_validate_connection(self.hass, self.client_config)

                if connect_err:
                    errors["base"] = f"failed_to_connect"
                    description_placeholders["exception"] = str(connect_err)
                else:
                    return await self._step_finish_async(user_input)
            
        return self.async_show_form(
            step_id="user", 
            data_schema=ui_schema_backend_connections(
                vector_db_backend_type=self.client_config[CONF_VECTOR_DB_BACKEND_TYPE],
                embedding_backend_type=self.client_config[CONF_EMBEDDING_BACKEND_TYPE],
                llm_backend_type=self.client_config[CONF_LLM_BACKEND_TYPE],
                vector_db_host=self.client_config.get(CONF_VECTOR_DB_HOST),
                vector_db_port=self.client_config.get(CONF_VECTOR_DB_PORT),
                vector_db_ssl=self.client_config.get(CONF_VECTOR_DB_SSL),
                vector_db_name=self.client_config.get(CONF_VECTOR_DB_NAME),
                embedding_host=self.client_config.get(CONF_EMBEDDING_HOST),
                embedding_port=self.client_config.get(CONF_EMBEDDING_PORT),
                embedding_ssl=self.client_config.get(CONF_EMBEDDING_SSL),
                llm_host=self.client_config.get(CONF_LLM_HOST),
                llm_port=self.client_config.get(CONF_LLM_PORT),
                llm_ssl=self.client_config.get(CONF_LLM_SSL)), 
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=True
        )
        
    async def _step_finish_async(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        language = self.client_config[CONF_SELECTED_LANGUAGE]
        vector_db_backend = self.client_config[CONF_VECTOR_DB_BACKEND_TYPE]
        embedding_backend = self.client_config[CONF_EMBEDDING_BACKEND_TYPE]
        llm_backend = self.client_config[CONF_LLM_BACKEND_TYPE]

        title = vector_db_to_class(vector_db_backend).get_name()
        title += " | " + embedding_backend_to_class(embedding_backend).get_name()
        title += " | " + llm_backend_to_class(llm_backend).get_name()
        title += " | Language: " + self.client_config.get(CONF_SELECTED_LANGUAGE, "en") 

        return self.async_create_entry(
            title=title,
            description="A local RAG agent.",
            data={
                CONF_SELECTED_LANGUAGE: language,
                CONF_VECTOR_DB_BACKEND_TYPE: vector_db_backend,
                CONF_EMBEDDING_BACKEND_TYPE: embedding_backend,
                CONF_LLM_BACKEND_TYPE: llm_backend
            },
            options=self.client_config,
        )
    
    @classmethod
    def async_supports_options_flow(cls, config_entry: RAGentConfigEntry) -> bool:
        return True

    @staticmethod
    def async_get_options_flow(config_entry: RAGentConfigEntry) -> OptionsFlow:
        return RagentOptionsFlow()
    
    @classmethod
    def async_get_supported_subentry_types(cls, config_entry: RAGentConfigEntry) -> dict[str, type[ConfigSubentryFlow]]:
        return { "ragent": RagentSubentryFlowHandler }
