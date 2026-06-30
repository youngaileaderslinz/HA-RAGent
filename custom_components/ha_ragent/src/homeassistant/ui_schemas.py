import logging
from typing import Any
import voluptuous as vol
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.data_entry_flow import AbortFlow
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TemplateSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    BooleanSelector,
    BooleanSelectorConfig,
)

from ..const import (
    BACKEND_VECTOR_DB_TYPE_FAISS,
    BACKEND_VECTOR_DB_TYPE_OPTIONS,
    CONF_NUM_TOOLS_TO_EXTRACT,
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_VECTOR_DB_USERNAME,
    CONF_VECTOR_DB_PASSWORD,
    CONF_VECTOR_DB_NAME,
    BACKEND_EMBEDDING_TYPE_OPTIONS,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_EMBEDDING_MODEL,
    BACKEND_LLM_TYPE_OPTIONS,
    CONF_LLM_BACKEND_TYPE,
    CONF_LLM_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_SELECTED_LANGUAGE,
    CONF_ENABLE_MODEL_THINKING,
    CONF_TEMPERATURE,
    CONF_K_TOP,
    CONF_P_MIN,
    CONF_P_TOP,
    CONF_P_TYPICAL,
    CONF_NUM_DEVICES_TO_EXTRACT,
    
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_VECTOR_DB_HOST,
    CONF_LLM_HOST,
    CONF_EMBEDDING_HOST,

    DEFAULT_EMBEDDING_BACKEND_TYPE,
    DEFAULT_LLM_BACKEND_TYPE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_NUM_TOOLS_TO_EXTRACT,
    DEFAULT_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_SELECTED_LANGUAGE,
    DEFAULT_TEMPERATURE,
    DEFAULT_K_TOP,
    DEFAULT_P_MIN,
    DEFAULT_P_TOP,
    DEFAULT_P_TYPICAL,
    DEFAULT_ENABLE_MODEL_THINKING,
    DEFAULT_VECTOR_DB_BACKEND_TYPE,
    DEFAULT_VECTOR_DB_BACKEND_TYPE,
    DEFAULT_VECTOR_DB_NAME,
    DEFAULT_NUM_DEVICES_TO_EXTRACT,
    
    BACKEND_VECTOR_DB_TYPE_MONGODB,
    BACKEND_VECTOR_DB_TYPE_CHROMA,

    SELECTED_LANGUAGE_OPTIONS,
)

from ..utils import (
    get_value
)

from .ragent import RAGent

_logger = logging.getLogger(__name__)

def ui_schema_pick_backends(ventor_db_backend_type=None, embedding_backend_type=None, llm_backend_type=None, selected_language=None) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(
                CONF_VECTOR_DB_BACKEND_TYPE,
                default=get_value(ventor_db_backend_type, DEFAULT_VECTOR_DB_BACKEND_TYPE)
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_VECTOR_DB_TYPE_OPTIONS,
                translation_key=CONF_VECTOR_DB_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_EMBEDDING_BACKEND_TYPE,
                default=get_value(embedding_backend_type, DEFAULT_EMBEDDING_BACKEND_TYPE)
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_EMBEDDING_TYPE_OPTIONS,
                translation_key=CONF_EMBEDDING_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_LLM_BACKEND_TYPE,
                default=get_value(llm_backend_type, DEFAULT_LLM_BACKEND_TYPE)
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_LLM_TYPE_OPTIONS,
                translation_key=CONF_LLM_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_SELECTED_LANGUAGE, 
                default=get_value(selected_language, DEFAULT_SELECTED_LANGUAGE)
            ): SelectSelector(SelectSelectorConfig(
                options=SELECTED_LANGUAGE_OPTIONS,
                translation_key=CONF_SELECTED_LANGUAGE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )

def ui_schema_backend_connections(
        vector_db_backend_type: str,
        embedding_backend_type: str, 
        llm_backend_type: str,
        vector_db_username=None,
        vector_db_password=None, 
        vector_db_host=None,
        vector_db_port=None,
        vector_db_ssl=None,
        vector_db_name=None,
        embedding_host=None, 
        embedding_port=None, 
        embedding_ssl=None,
        llm_host=None,
        llm_port=None,
        llm_ssl=None) -> vol.Schema:
    if vector_db_backend_type not in BACKEND_VECTOR_DB_TYPE_OPTIONS:
        raise AbortFlow(reason="unknown_vector_db_backend_type")
    
    if embedding_backend_type not in BACKEND_EMBEDDING_TYPE_OPTIONS:
        raise AbortFlow(reason="unknown_embedding_backend_type")

    if llm_backend_type not in BACKEND_LLM_TYPE_OPTIONS:
        raise AbortFlow(reason="unknown_llm_backend_type")
    
    default_port_mongodb = 27017
    default_port_chroma = 8000
    default_port_ollama = 11434

    if vector_db_backend_type == BACKEND_VECTOR_DB_TYPE_MONGODB:
        vector_default_port = default_port_mongodb
    elif vector_db_backend_type == BACKEND_VECTOR_DB_TYPE_CHROMA:
        vector_default_port = default_port_chroma

    schema = {}
    
    if vector_db_backend_type == BACKEND_VECTOR_DB_TYPE_MONGODB:
        schema.update({
            vol.Optional(CONF_VECTOR_DB_USERNAME, default=vector_db_username if vector_db_username else ""): str,
            vol.Optional(CONF_VECTOR_DB_PASSWORD, default=vector_db_password if vector_db_password else ""): str,
        })
        
    if not vector_db_backend_type == BACKEND_VECTOR_DB_TYPE_FAISS:
        schema.update({
            vol.Required(CONF_VECTOR_DB_HOST, default=vector_db_host if vector_db_host else ""): str,
            vol.Optional(CONF_VECTOR_DB_PORT, default=vector_db_port if vector_db_port else vector_default_port): int,
            vol.Required(CONF_VECTOR_DB_SSL, default=vector_db_ssl if vector_db_ssl else False): bool,
        })
    
    schema.update({
        vol.Required(CONF_VECTOR_DB_NAME, default=vector_db_name if vector_db_name else f"{DEFAULT_VECTOR_DB_NAME}_{uuid4()}"): str,

        vol.Required(CONF_EMBEDDING_HOST, default=embedding_host if embedding_host else ""): str,
        vol.Optional(CONF_EMBEDDING_PORT, default=embedding_port if embedding_port else default_port_ollama): int,
        vol.Required(CONF_EMBEDDING_SSL, default=embedding_ssl if embedding_ssl else False): bool,

        vol.Required(CONF_LLM_HOST, default=llm_host if llm_host else ""): str,
        vol.Optional(CONF_LLM_PORT, default=llm_port if llm_port else default_port_ollama): int,
        vol.Required(CONF_LLM_SSL, default=llm_ssl if llm_ssl else False): bool
    })

    return vol.Schema(schema)

def ui_schema_pick_models(embedding_models: list[str], llm_models: list[str], embedding_model: str | None = None, llm_model: str | None = None) -> vol.Schema:
    if len(embedding_models) == 0:
        embedding_models = [ "" ]
    if len(llm_models) == 0:
        llm_models = [ "" ]
    
    return vol.Schema(
        {
            vol.Required(CONF_EMBEDDING_MODEL, default=embedding_model if embedding_model else embedding_models[0]): SelectSelector(SelectSelectorConfig(
                options=embedding_models,
                custom_value=False,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),

            vol.Required(CONF_LLM_MODEL, default=llm_model if llm_model else llm_models[0]): SelectSelector(SelectSelectorConfig(
                options=llm_models,
                custom_value=False,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )


def ui_schema_config_options(
    hass: HomeAssistant,
    language: str,
    options: dict[str, Any],
    vector_db_backend_type: str,
    embedding_backend_type: str,
    llm_backend_type: str, 
    subentry_type: str,
) -> dict:
    default_prompt = RAGent.build_base_prompt_template(language, DEFAULT_PROMPT)

    llm_api_options = [SelectOptionDict(value="none", label="No Control")]
    try:
        for api in llm.async_get_apis(hass):
            api_label = getattr(api, "name", None) or api.id
            llm_api_options.append(SelectOptionDict(value=api.id, label=str(api_label)))
    except Exception as err:
        _logger.warning("Failed to load LLM APIs: %s", err)

    result: dict = {
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API, "none")},
            default=options.get(CONF_LLM_HASS_API, "none"),
        ): SelectSelector(SelectSelectorConfig(
            options=llm_api_options,
            custom_value=False,
            multiple=False,
            mode=SelectSelectorMode.DROPDOWN,
        )),
        vol.Optional(
            CONF_PROMPT,
            default=options.get(CONF_PROMPT, default_prompt),
        ): TextSelector(TextSelectorConfig(
            multiline=True,
            type=TextSelectorType.TEXT,
        )),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)},
            default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        ): NumberSelector(NumberSelectorConfig(min=0.0, max=2.0, step=0.05, mode=NumberSelectorMode.BOX)),
        vol.Required(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=DEFAULT_MAX_TOKENS,
        ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
        vol.Required(
            CONF_CONTEXT_LENGTH,
            description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
            default=DEFAULT_CONTEXT_LENGTH,
        ): NumberSelector(NumberSelectorConfig(min=512, max=1_048_576, step=512)),
        # vol.Required(
        #     CONF_K_TOP,
        #     description={"suggested_value": options.get(CONF_K_TOP)},
        #     default=DEFAULT_K_TOP,
        # ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
        # vol.Required(
        #     CONF_P_TOP,
        #     description={"suggested_value": options.get(CONF_P_TOP)},
        #     default=DEFAULT_P_TOP,
        # ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        #  vol.Required(
        #     CONF_P_MIN,
        #     description={"suggested_value": options.get(CONF_P_MIN)},
        #     default=DEFAULT_P_MIN,
        # ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        # vol.Required(
        #     CONF_P_TYPICAL,
        #     description={"suggested_value": options.get(CONF_P_TYPICAL)},
        #     default=DEFAULT_P_TYPICAL,
        # ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
            description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS)},
            default=options.get(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS),
        ): NumberSelector(NumberSelectorConfig(min=0, max=100, mode=NumberSelectorMode.BOX)),
        vol.Optional(
            CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
            description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES)},
            default=options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES),
        ): NumberSelector(NumberSelectorConfig(min=0, max=1440, mode=NumberSelectorMode.BOX)),
        vol.Required(
            CONF_MAX_TOOL_CALL_ITERATIONS,
            description={"suggested_value": options.get(CONF_MAX_TOOL_CALL_ITERATIONS)},
            default=DEFAULT_MAX_TOOL_CALL_ITERATIONS,
        ): int,
        vol.Optional(
            CONF_ENABLE_MODEL_THINKING,
            description={"suggested_value": options.get(CONF_ENABLE_MODEL_THINKING, DEFAULT_ENABLE_MODEL_THINKING)},
            default=options.get(CONF_ENABLE_MODEL_THINKING, DEFAULT_ENABLE_MODEL_THINKING),
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_NUM_DEVICES_TO_EXTRACT,
            description={"suggested_value": options.get(CONF_NUM_DEVICES_TO_EXTRACT)},
            default=DEFAULT_NUM_DEVICES_TO_EXTRACT,
        ): int,
        vol.Required(
            CONF_NUM_TOOLS_TO_EXTRACT,
            description={"suggested_value": options.get(CONF_NUM_TOOLS_TO_EXTRACT)},
            default=DEFAULT_NUM_TOOLS_TO_EXTRACT,
        ): int,
    }

    global_order = [
        # general
        CONF_LLM_HASS_API,
        CONF_PROMPT,
        CONF_ENABLE_MODEL_THINKING,
        CONF_NUM_DEVICES_TO_EXTRACT,
        CONF_NUM_TOOLS_TO_EXTRACT,
        CONF_CONTEXT_LENGTH,
        CONF_MAX_TOKENS,
        # sampling parameters
        CONF_TEMPERATURE,
        CONF_P_TOP,
        CONF_P_MIN,
        CONF_P_TYPICAL,
        CONF_K_TOP,
        # tool and memory parameters
        CONF_MAX_TOOL_CALL_ITERATIONS,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    ]

    result = { k: v for k, v in sorted(result.items(), key=lambda item: global_order.index(item[0]) if item[0] in global_order else 9999) }

    return vol.Schema(result)
