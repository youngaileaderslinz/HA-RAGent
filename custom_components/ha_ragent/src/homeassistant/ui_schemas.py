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
    BooleanSelector,
    BooleanSelectorConfig,
)

from custom_components.ha_ragent.src.const import (
    BACKEND_VECTOR_DB_TYPE_FAISS,
    BACKEND_VECTOR_DB_TYPE_OPTIONS,
    CONF_NUM_TOOLS_TO_EXTRACT,
    CONF_NUM_MEMORIES_TO_EXTRACT,
    CONF_MAX_MEMORY_ENTRIES,
    CONF_EXCLUDED_TOOLS,
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
    RAGENT_LLM_API_ID,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_OPTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_SELECTED_LANGUAGE,
    CONF_ENABLE_MODEL_THINKING,
    CONF_ALLOW_AUTO_EMBEDDING,
    CONF_ALLOW_QUESTIONS,
    CONF_TEMPERATURE,
    CONF_K_TOP,
    CONF_P_MIN,
    CONF_P_TOP,
    CONF_P_TYPICAL,
    CONF_NUM_DEVICES_TO_EXTRACT,
    DEFAULT_PROMPT,
    
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_EMBEDDING_API_KEY,
    CONF_VECTOR_DB_HOST,
    CONF_LLM_HOST,
    CONF_EMBEDDING_HOST,
    CONF_LLM_API_KEY,
    CONF_RETRIEVAL_METHOD,

    
    BACKEND_VECTOR_DB_TYPE_MONGODB,
    BACKEND_VECTOR_DB_TYPE_CHROMA,
    BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE,
    BACKEND_LLM_TYPE_OPENAI_COMPATIBLE,
    EMBEDDING_BACKENDS_WITH_API_KEY,
    LLM_BACKENDS_WITH_API_KEY,

    SELECTED_LANGUAGE_OPTIONS,
)

from custom_components.ha_ragent.src.utils import get_value, get_setting_value

from custom_components.ha_ragent.src.homeassistant.ragent import RAGent

_logger = logging.getLogger(__name__)


def _backend_connection_defaults(backend_type: str, *, ollama_port: int, openai_port: int) -> tuple[int, bool]:
    if backend_type in (BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE, BACKEND_LLM_TYPE_OPENAI_COMPATIBLE):
        return openai_port, True

    return ollama_port, False

def ui_schema_pick_backends(ventor_db_backend_type=None, embedding_backend_type=None, llm_backend_type=None, selected_language=None) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(
                CONF_VECTOR_DB_BACKEND_TYPE,
                default=get_value(ventor_db_backend_type, get_setting_value(CONF_VECTOR_DB_BACKEND_TYPE, options))
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_VECTOR_DB_TYPE_OPTIONS,
                translation_key=CONF_VECTOR_DB_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_EMBEDDING_BACKEND_TYPE,
                default=get_value(embedding_backend_type, get_setting_value(CONF_EMBEDDING_BACKEND_TYPE, options))
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_EMBEDDING_TYPE_OPTIONS,
                translation_key=CONF_EMBEDDING_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_LLM_BACKEND_TYPE,
                default=get_value(llm_backend_type, get_setting_value(CONF_LLM_BACKEND_TYPE, options))
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_LLM_TYPE_OPTIONS,
                translation_key=CONF_LLM_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_SELECTED_LANGUAGE, 
                default=get_value(selected_language, get_setting_value(CONF_SELECTED_LANGUAGE, options))
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
        embedding_api_key=None,
        llm_host=None,
        llm_port=None,
        llm_ssl=None,
        llm_api_key=None) -> vol.Schema:
    if vector_db_backend_type not in BACKEND_VECTOR_DB_TYPE_OPTIONS:
        raise AbortFlow(reason="unknown_vector_db_backend_type")
    
    if embedding_backend_type not in BACKEND_EMBEDDING_TYPE_OPTIONS:
        raise AbortFlow(reason="unknown_embedding_backend_type")

    if llm_backend_type not in BACKEND_LLM_TYPE_OPTIONS:
        raise AbortFlow(reason="unknown_llm_backend_type")
    
    default_port_mongodb = 27017
    default_port_chroma = 8000
    default_port_ollama = 11434
    default_port_openai = 443

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

    embedding_default_port, embedding_default_ssl = _backend_connection_defaults(
        embedding_backend_type,
        ollama_port=default_port_ollama,
        openai_port=default_port_openai,
    )
    llm_default_port, llm_default_ssl = _backend_connection_defaults(
        llm_backend_type,
        ollama_port=default_port_ollama,
        openai_port=default_port_openai,
    )
    
    schema.update({
        vol.Required(CONF_VECTOR_DB_NAME, default=vector_db_name if vector_db_name else f"ha_ragent_db_{uuid4()}"): str,
    })

    schema.update({
        vol.Required(CONF_EMBEDDING_HOST, default=embedding_host if embedding_host else ""): str,
        vol.Optional(CONF_EMBEDDING_PORT, default=embedding_port if embedding_port else embedding_default_port): int,
    })

    if embedding_backend_type in EMBEDDING_BACKENDS_WITH_API_KEY:
        schema[vol.Optional(CONF_EMBEDDING_API_KEY, default=embedding_api_key if embedding_api_key else "")] = str
    schema[vol.Required(CONF_EMBEDDING_SSL, default=embedding_ssl if embedding_ssl is not None else embedding_default_ssl)] = bool

    schema.update({
        vol.Required(CONF_LLM_HOST, default=llm_host if llm_host else ""): str,
        vol.Optional(CONF_LLM_PORT, default=llm_port if llm_port else llm_default_port): int,
    })

    if llm_backend_type in LLM_BACKENDS_WITH_API_KEY:
        schema[vol.Optional(CONF_LLM_API_KEY, default=llm_api_key if llm_api_key else "")] = str
    schema[vol.Required(CONF_LLM_SSL, default=llm_ssl if llm_ssl is not None else llm_default_ssl)] = bool

    return vol.Schema(schema)

def ui_schema_pick_models(
    embedding_models: list[str],
    llm_models: list[str],
    embedding_model: str | None = None,
    llm_model: str | None = None,
) -> vol.Schema:
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
    excluded_tool_options: list[str] | None = None,
) -> dict:
    default_prompt = RAGent.build_base_prompt_template(
        language,
        get_setting_value(CONF_PROMPT, options) or DEFAULT_PROMPT,
    )
    default_llm_api = getattr(llm, "LLM_API_ASSIST", "assist")
    selected_llm_api = options.get(CONF_LLM_HASS_API, default_llm_api)

    llm_api_options = [SelectOptionDict(value="none", label="No Control")]
    try:
        for api in llm.async_get_apis(hass):
            if api.id == RAGENT_LLM_API_ID:
                continue
            api_label = getattr(api, "name", None) or api.id
            llm_api_options.append(SelectOptionDict(value=api.id, label=str(api_label)))
    except Exception as err:
        _logger.warning("Failed to load LLM APIs: %s", err)

    result: dict = {
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": selected_llm_api},
            default=selected_llm_api,
        ): SelectSelector(SelectSelectorConfig(
            options=llm_api_options,
            custom_value=False,
            multiple=False,
            mode=SelectSelectorMode.DROPDOWN,
        )),
        vol.Optional(
            CONF_PROMPT,
            default=options.get(CONF_PROMPT, default_prompt),
        ): TemplateSelector(),
        vol.Required(
            CONF_RETRIEVAL_METHOD,
            description={"suggested_value": options.get(CONF_RETRIEVAL_METHOD, RETRIEVAL_METHOD_OPTIONS[0])},
            default=options.get(CONF_RETRIEVAL_METHOD, RETRIEVAL_METHOD_OPTIONS[0]),
        ): SelectSelector(SelectSelectorConfig(
            options=[
                SelectOptionDict(value="automatic", label="Automatic"),
                SelectOptionDict(value="vector", label="Vector search"),
                SelectOptionDict(value="lexical", label="Lexical search"),
            ],
            custom_value=False,
            multiple=False,
            mode=SelectSelectorMode.DROPDOWN,
        )),
        vol.Optional(
            CONF_ALLOW_AUTO_EMBEDDING,
            description={"suggested_value": get_setting_value(CONF_ALLOW_AUTO_EMBEDDING, options)},
            default=get_setting_value(CONF_ALLOW_AUTO_EMBEDDING, options),
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Optional(
            CONF_ALLOW_QUESTIONS,
            description={"suggested_value": get_setting_value(CONF_ALLOW_QUESTIONS, options)},
            default=get_setting_value(CONF_ALLOW_QUESTIONS, options),
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": get_setting_value(CONF_TEMPERATURE, options)},
            default=get_setting_value(CONF_TEMPERATURE, options),
        ): NumberSelector(NumberSelectorConfig(min=0.0, max=2.0, step=0.05, mode=NumberSelectorMode.BOX)),
        vol.Required(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=get_setting_value(CONF_MAX_TOKENS, options),
        ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
        vol.Required(
            CONF_CONTEXT_LENGTH,
            description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
            default=get_setting_value(CONF_CONTEXT_LENGTH, options),
        ): NumberSelector(NumberSelectorConfig(min=512, max=1_048_576, step=512)),
        # vol.Required(
        #     CONF_K_TOP,
        #     description={"suggested_value": options.get(CONF_K_TOP)},
        # ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
        # vol.Required(
        #     CONF_P_TOP,
        #     description={"suggested_value": options.get(CONF_P_TOP)},
        # ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        #  vol.Required(
        #     CONF_P_MIN,
        #     description={"suggested_value": options.get(CONF_P_MIN)},
        # ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        # vol.Required(
        #     CONF_P_TYPICAL,
        #     description={"suggested_value": options.get(CONF_P_TYPICAL)},
        # ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
            description={"suggested_value": get_setting_value(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, options)},
            default=get_setting_value(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, options),
        ): NumberSelector(NumberSelectorConfig(min=0, max=100, mode=NumberSelectorMode.BOX)),
        vol.Optional(
            CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
            description={"suggested_value": get_setting_value(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, options)},
            default=get_setting_value(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, options),
        ): NumberSelector(NumberSelectorConfig(min=0, max=1440, mode=NumberSelectorMode.BOX)),
        vol.Required(
            CONF_MAX_TOOL_CALL_ITERATIONS,
            description={"suggested_value": options.get(CONF_MAX_TOOL_CALL_ITERATIONS)},
            default=get_setting_value(CONF_MAX_TOOL_CALL_ITERATIONS, options),
        ): int,
        vol.Optional(
            CONF_ENABLE_MODEL_THINKING,
            description={"suggested_value": get_setting_value(CONF_ENABLE_MODEL_THINKING, options)},
            default=get_setting_value(CONF_ENABLE_MODEL_THINKING, options),
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_NUM_DEVICES_TO_EXTRACT,
            description={"suggested_value": options.get(CONF_NUM_DEVICES_TO_EXTRACT)},
            default=get_setting_value(CONF_NUM_DEVICES_TO_EXTRACT, options),
        ): int,
        vol.Required(
            CONF_NUM_TOOLS_TO_EXTRACT,
            description={"suggested_value": options.get(CONF_NUM_TOOLS_TO_EXTRACT)},
            default=get_setting_value(CONF_NUM_TOOLS_TO_EXTRACT, options),
        ): int,
        vol.Optional(
            CONF_NUM_MEMORIES_TO_EXTRACT,
            description={"suggested_value": get_setting_value(CONF_NUM_MEMORIES_TO_EXTRACT, options)},
            default=get_setting_value(CONF_NUM_MEMORIES_TO_EXTRACT, options),
        ): NumberSelector(NumberSelectorConfig(min=0, max=20, mode=NumberSelectorMode.BOX)),
        vol.Optional(
            CONF_MAX_MEMORY_ENTRIES,
            description={"suggested_value": get_setting_value(CONF_MAX_MEMORY_ENTRIES, options)},
            default=get_setting_value(CONF_MAX_MEMORY_ENTRIES, options),
        ): NumberSelector(NumberSelectorConfig(min=1, max=10000, mode=NumberSelectorMode.BOX)),
        vol.Optional(
            CONF_EXCLUDED_TOOLS,
            default=options.get(CONF_EXCLUDED_TOOLS, []),
        ): SelectSelector(SelectSelectorConfig(
            options=[SelectOptionDict(value=name, label=name) for name in sorted(excluded_tool_options or [])],
            custom_value=True,
            multiple=True,
            mode=SelectSelectorMode.DROPDOWN,
        )),
    }

    if llm_backend_type == BACKEND_LLM_TYPE_OPENAI_COMPATIBLE:
        result.pop(CONF_CONTEXT_LENGTH, None)

    global_order = [
        # general
        CONF_LLM_HASS_API,
        CONF_PROMPT,
        CONF_ALLOW_AUTO_EMBEDDING,
        CONF_ALLOW_QUESTIONS,
        CONF_ENABLE_MODEL_THINKING,
        CONF_RETRIEVAL_METHOD,
        CONF_NUM_DEVICES_TO_EXTRACT,
        CONF_NUM_TOOLS_TO_EXTRACT,
        CONF_NUM_MEMORIES_TO_EXTRACT,
        CONF_MAX_MEMORY_ENTRIES,
        CONF_EXCLUDED_TOOLS,
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
