from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from typing import Any

from homeassistant.config_entries import (
    ConfigFlowResult,
    OptionsFlow,
)

from custom_components.ha_ragent.src.const import (
    CONF_LLM_BACKEND_TYPE,
    CONF_LLM_HOST,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_LLM_API_KEY,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_EMBEDDING_API_KEY,
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
    CONF_VECTOR_DB_PASSWORD,
    CONF_VECTOR_DB_USERNAME
)

from custom_components.ha_ragent.src.backends.backends import embedding_backend_to_class, llm_backend_to_class, vector_db_to_class
from custom_components.ha_ragent.src.homeassistant.ui_schemas import ui_schema_backend_connections

_logger = BaseLogger(__name__)

class RagentOptionsFlow(OptionsFlow):
    def __init__(self):
        super().__init__()
        self.model_config: dict[str, Any] | None = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors = {}
        description_placeholders = {}
        client_config = dict(self.config_entry.options)

        if user_input is not None:
            client_config.update(user_input)
            connect_err = await vector_db_to_class(client_config.get(CONF_VECTOR_DB_BACKEND_TYPE)).async_validate_connection(self.hass, client_config)

            if not connect_err:
                connect_err = await embedding_backend_to_class(client_config.get(CONF_EMBEDDING_BACKEND_TYPE)).async_validate_connection(self.hass, client_config)
            
            if not connect_err:
                connect_err = await llm_backend_to_class(client_config.get(CONF_LLM_BACKEND_TYPE)).async_validate_connection(self.hass, client_config)

            if not connect_err:
                return self.async_create_entry(data=client_config)
            else:
                errors["base"] = "failed_to_connect"
                description_placeholders["exception"] = str(connect_err)

        schema = ui_schema_backend_connections(
            vector_db_backend_type=client_config.get(CONF_VECTOR_DB_BACKEND_TYPE),
            embedding_backend_type=client_config.get(CONF_EMBEDDING_BACKEND_TYPE),
            llm_backend_type=client_config.get(CONF_LLM_BACKEND_TYPE),
            vector_db_username=client_config.get(CONF_VECTOR_DB_USERNAME),
            vector_db_password=client_config.get(CONF_VECTOR_DB_PASSWORD),
            vector_db_host=client_config.get(CONF_VECTOR_DB_HOST),
            vector_db_port=client_config.get(CONF_VECTOR_DB_PORT),
            vector_db_ssl=client_config.get(CONF_VECTOR_DB_SSL),
            embedding_host=client_config.get(CONF_EMBEDDING_HOST),
            embedding_port=client_config.get(CONF_EMBEDDING_PORT),
            embedding_ssl=client_config.get(CONF_EMBEDDING_SSL),
            embedding_api_key=client_config.get(CONF_EMBEDDING_API_KEY),
            llm_host=client_config.get(CONF_LLM_HOST),
            llm_port=client_config.get(CONF_LLM_PORT),
            llm_ssl=client_config.get(CONF_LLM_SSL),
            llm_api_key=client_config.get(CONF_LLM_API_KEY))
        
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )