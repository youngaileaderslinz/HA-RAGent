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
from custom_components.ha_ragent.src.utils import get_setting_value

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
            connect_err = await vector_db_to_class(get_setting_value(CONF_VECTOR_DB_BACKEND_TYPE, client_config)).async_validate_connection(self.hass, client_config)

            if not connect_err:
                connect_err = await embedding_backend_to_class(get_setting_value(CONF_EMBEDDING_BACKEND_TYPE, client_config)).async_validate_connection(self.hass, client_config)
            
            if not connect_err:
                connect_err = await llm_backend_to_class(get_setting_value(CONF_LLM_BACKEND_TYPE, client_config)).async_validate_connection(self.hass, client_config)

            if not connect_err:
                return self.async_create_entry(data=client_config)
            else:
                errors["base"] = "failed_to_connect"
                description_placeholders["exception"] = str(connect_err)

        schema = ui_schema_backend_connections(
            vector_db_backend_type=get_setting_value(CONF_VECTOR_DB_BACKEND_TYPE, client_config),
            embedding_backend_type=get_setting_value(CONF_EMBEDDING_BACKEND_TYPE, client_config),
            llm_backend_type=get_setting_value(CONF_LLM_BACKEND_TYPE, client_config),
            vector_db_username=get_setting_value(CONF_VECTOR_DB_USERNAME, client_config),
            vector_db_password=get_setting_value(CONF_VECTOR_DB_PASSWORD, client_config),
            vector_db_host=get_setting_value(CONF_VECTOR_DB_HOST, client_config),
            vector_db_port=get_setting_value(CONF_VECTOR_DB_PORT, client_config),
            vector_db_ssl=get_setting_value(CONF_VECTOR_DB_SSL, client_config),
            embedding_host=get_setting_value(CONF_EMBEDDING_HOST, client_config),
            embedding_port=get_setting_value(CONF_EMBEDDING_PORT, client_config),
            embedding_ssl=get_setting_value(CONF_EMBEDDING_SSL, client_config),
            embedding_api_key=get_setting_value(CONF_EMBEDDING_API_KEY, client_config),
            llm_host=get_setting_value(CONF_LLM_HOST, client_config),
            llm_port=get_setting_value(CONF_LLM_PORT, client_config),
            llm_ssl=get_setting_value(CONF_LLM_SSL, client_config),
            llm_api_key=get_setting_value(CONF_LLM_API_KEY, client_config))
        
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )