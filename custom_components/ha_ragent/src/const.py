import re

PLATFORMS = ("conversation",)
CONFIG_FLOW_VERSION = 1

#-----------------------------------------------
# General constants
#-----------------------------------------------
DOMAIN = "ha_ragent"
RAGENT_LLM_API_ID = "ha_ragent_api"
RAGENT_LLM_API_NAME = "HA-RAGent"
HOME_ASSISTANT_SCRIPT_DOMAIN = "script"
STARTUP_EMBEDDING_RUNNING_FLAG = "ha_ragent_startup_embedding_running"

CONNECTION_RETRIES = 3
STREAM_READ_TIMEOUT = 400
HTTP_REQUEST_TIMEOUT = 5
STREAM_CONNECT_TIMEOUT = 30
RETRY_BACKOFF_BASE_SECONDS = 0.5
RETRY_BACKOFF_MULTIPLIER = 2

CONF_LLM_HASS_API = "llm_hass_api"

RAGENT_SEMANTIC_SEARCH_TOOL_NAME = "HassSemanticSearch"
RAGENT_PLANNED_ACTION_TOOL_NAME = "HassPlannedAction"
RAGENT_CANCEL_ALL_PLANNED_ACTIONS_TOOL_NAME = "HassCancelAllPlannedActions"
RAGENT_LIST_PLANNED_ACTIONS_TOOL_NAME = "HassListPlannedActions"
RAGENT_REMEMBER_TOOL_NAME = "HassRememberFact"
RAGENT_FORGET_TOOL_NAME = "HassForgetFact"
RAGENT_SCHEDULED_ACTION_CANCELLERS = "scheduled_action_cancellers"
RAGENT_SCHEDULED_ACTIONS = "scheduled_actions"
RAGENT_MEMORY_LOCKS = "memory_locks"
RAGENT_SCHEDULED_REQUEST_PREFIX = "[scheduled-action] "
RAGENT_SCHEDULED_CONTEXT_PREFIX = "[scheduled-context:"
RAGENT_SCHEDULED_EXECUTION_CONTEXTS = "scheduled_execution_contexts"

RAGENT_TOOL_NAMES = [
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
    RAGENT_PLANNED_ACTION_TOOL_NAME,
    RAGENT_CANCEL_ALL_PLANNED_ACTIONS_TOOL_NAME,
    RAGENT_LIST_PLANNED_ACTIONS_TOOL_NAME,
    RAGENT_REMEMBER_TOOL_NAME,
    RAGENT_FORGET_TOOL_NAME,
]
RAGENT_PREFIXED_TOOL_NAMES = [ f"{DOMAIN}__{tool_name}" for tool_name in RAGENT_TOOL_NAMES ]
RAGENT_PREFIXED_TOOL_NAMES_BY_NAME = dict(zip(RAGENT_TOOL_NAMES, RAGENT_PREFIXED_TOOL_NAMES))
RAGENT_TOOL_NAMES_BY_PREFIXED_NAME = { prefixed_name: tool_name for tool_name, prefixed_name in RAGENT_PREFIXED_TOOL_NAMES_BY_NAME.items() }

RAGENT_REQUIRED_TOOL_NAMES = [ RAGENT_SEMANTIC_SEARCH_TOOL_NAME ]
RAGENT_PREFIXED_REQUIRED_TOOL_NAMES = [ RAGENT_PREFIXED_TOOL_NAMES_BY_NAME[name] for name in RAGENT_REQUIRED_TOOL_NAMES ]

RAGENT_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES = [
    RAGENT_PLANNED_ACTION_TOOL_NAME,
    RAGENT_LIST_PLANNED_ACTIONS_TOOL_NAME,
    RAGENT_CANCEL_ALL_PLANNED_ACTIONS_TOOL_NAME,
    RAGENT_REMEMBER_TOOL_NAME,
    RAGENT_FORGET_TOOL_NAME,
]
RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES = [
    RAGENT_PREFIXED_TOOL_NAMES_BY_NAME[name]
    for name in RAGENT_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES
]


#-----------------------------------------------
# Language constants
#-----------------------------------------------
CONF_SELECTED_LANGUAGE = "selected_language"

SELECTED_LANGUAGE_OPTIONS = [ 
    "en", 
    "de" 
]

TRANSLATION_PROMPT_PERSONA = "PERSONA_PROMPTS"
TRANSLATION_PROMPT_AREAS = "AREAS_PROMPT"
TRANSLATION_PROMPT_DEVICES = "DEVICES_PROMPT"
TRANSLATION_PROMPT_MEMORIES = "MEMORIES_CONTEXT_PROMPT"
TRANSLATION_PROMPT_RETRIES = "MAX_RETRIES_PROMPT"
TRANSLATION_PROMPT_SCHEDULED_ACTION = "SCHEDULED_ACTION_PROMPT"
TRANSLATION_PROMPT_INSTRUCTIONS = "INSTRUCTION_PROMPT"
TRANSLATION_PROMPT_SEARCH_FALLBACK = "SEARCH_FALLBACK_PROMPT"
TRANSLATION_ERROR_BACKEND = "backend_error"
TRANSLATION_ERROR_MAX_RETRIES = "max_retries_exhausted"
TRANSLATION_ERROR_LLM_API = "llm_api_error"
TRANSLATION_ERROR_TEMPLATE = "template_rendering_failed"
TRANSLATION_ERROR_UNEXPECTED = "unexpected"
TRANSLATION_ERROR_DESCRIPTION_EMPTY = "description_empty"
TRANSLATION_ERROR_MINUTES_NOT_NUMBER = "minutes_not_number"
TRANSLATION_ERROR_MINUTES_RANGE = "minutes_out_of_range"
TRANSLATION_ERROR_MEMORY_EMPTY = "memory_empty"
TRANSLATION_ERROR_MEMORY_TOO_LONG = "memory_too_long"
TRANSLATION_ERROR_MEMORY_STORE = "memory_store_failed"
TRANSLATION_ERROR_MEMORY_ID_INVALID = "memory_id_invalid"
TRANSLATION_ERROR_MEMORY_NOT_FOUND = "memory_not_found"
TRANSLATION_ERROR_SEARCH_QUERY_EMPTY = "search_query_empty"
TRANSLATION_ERROR_SEARCH_QUERIES_TOO_MANY = "search_queries_too_many"


#-----------------------------------------------
# Service Tool constants
#-----------------------------------------------
RAGENT_TIMER_DEVICE_ID = "ha_ragent_timer_device_a03a100a-81ca-415d"


#-----------------------------------------------
# Retrieval constants
#-----------------------------------------------
CONF_RETRIEVAL_METHOD = "rag_retrieval_method"

RETRIEVAL_METHOD_AUTOMATIC = "automatic"
RETRIEVAL_METHOD_VECTOR = "vector"
RETRIEVAL_METHOD_LEXICAL = "lexical"
RETRIEVAL_METHOD_OPTIONS = (
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_VECTOR,
    RETRIEVAL_METHOD_LEXICAL,
)

CANONICAL_NAME_SPLIT_PATTERN = r"_|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
RETRIEVAL_TOOL_SIGNAL_WEIGHTS = {
    "semantic_rank": 0.75,
    "semantic_similarity": 1.0,
    "lexical_exact": 2.0,
    "lexical_fuzzy": 0.75,
    "lexical_corpus": 3.0,
    "lexical_action": 4.0,
    "domain": 1.5,
    "device_metadata": 0.5,
    "continuity": 0.5,
}


#-----------------------------------------------
# Vector database backend constants
#-----------------------------------------------
CONF_VECTOR_DB_BACKEND_TYPE = "rag_vector_db_backend"
CONF_VECTOR_DB_NAME = "rag_vector_db_name"
CONF_VECTOR_DB_USERNAME = "rag_vector_db_username"
CONF_VECTOR_DB_PASSWORD = "rag_vector_db_password"
CONF_VECTOR_DB_HOST = "rag_vector_db_host"
CONF_VECTOR_DB_PORT = "rag_vector_db_port"
CONF_VECTOR_DB_SSL = "rag_vector_db_ssl"

BACKEND_VECTOR_DB_TYPE_MONGODB = "mongodb"
BACKEND_VECTOR_DB_TYPE_CHROMA = "chromadb"
BACKEND_VECTOR_DB_TYPE_FAISS = "faiss"

BACKEND_VECTOR_DB_TYPE_OPTIONS = [ 
    BACKEND_VECTOR_DB_TYPE_MONGODB,
    BACKEND_VECTOR_DB_TYPE_CHROMA,
    BACKEND_VECTOR_DB_TYPE_FAISS
]


#-----------------------------------------------
# Embedding backend constants
#-----------------------------------------------
RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS = 4000
RAGENT_EMBEDDING_TRUNCATE_RETRIES = 3
RAGENT_EMBEDDING_BATCH_SIZE = 32
CONF_EMBEDDING_BACKEND_TYPE = "rag_embedding_backend"
CONF_EMBEDDING_MODEL = "rag_embedding_model"
CONF_EMBEDDING_HOST = "rag_embedding_host"
CONF_EMBEDDING_PORT = "rag_embedding_port"
CONF_EMBEDDING_SSL = "rag_embedding_ssl"
CONF_EMBEDDING_API_KEY = "rag_embedding_api_key"

BACKEND_EMBEDDING_TYPE_OLLAMA = "ollama"
BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE = "openai_compatible"

BACKEND_EMBEDDING_TYPE_OPTIONS = [ 
    BACKEND_EMBEDDING_TYPE_OLLAMA,
    BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE,
]

EMBEDDING_BACKENDS_WITH_API_KEY = [ BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE ]


#-----------------------------------------------
# Chat backend constants
#-----------------------------------------------
RAGENT_CHAT_TRUNCATE_MAX_CHARS = 12000
RAGENT_CHAT_TRUNCATE_RETRIES = 3
RAGENT_MAX_SEARCH_QUERY_CHARS = 4000
RAGENT_MAX_SEARCH_QUERIES = 4

CONF_LLM_BACKEND_TYPE = "rag_llm_backend"
CONF_LLM_MODEL = "rag_llm_model"
CONF_LLM_HOST = "rag_llm_host"
CONF_LLM_PORT = "rag_llm_port"
CONF_LLM_SSL = "rag_llm_ssl"
CONF_LLM_API_KEY = "rag_llm_api_key"

BACKEND_LLM_TYPE_OLLAMA = "ollama"
BACKEND_LLM_TYPE_OPENAI_COMPATIBLE = "openai_compatible"

BACKEND_LLM_TYPE_OPTIONS = [ 
    BACKEND_LLM_TYPE_OLLAMA,
    BACKEND_LLM_TYPE_OPENAI_COMPATIBLE,
]

LLM_BACKENDS_WITH_API_KEY = [ BACKEND_LLM_TYPE_OPENAI_COMPATIBLE ]


#-----------------------------------------------
# Prompt configuration constants
#----------------------------------------------
CONF_NUM_DEVICES_TO_EXTRACT = "rag_num_devices_to_extract"
CONF_NUM_TOOLS_TO_EXTRACT = "rag_num_tools_to_extract"
CONF_NUM_MEMORIES_TO_EXTRACT = "rag_num_memories_to_extract"
CONF_MAX_MEMORY_ENTRIES = "rag_max_memory_entries"
CONF_EXCLUDED_TOOLS = "rag_excluded_tools"
CONF_CONTEXT_LENGTH = "rag_context_length"

CONF_MAX_TOKENS = "rag_max_tokens"
CONF_MAX_TOOL_CALL_ITERATIONS = "rag_max_tool_call_iterations"

CONF_PROMPT = "rag_prompt"

CONF_ENABLE_MODEL_THINKING = "rag_enable_model_thinking"
CONF_ALLOW_AUTO_EMBEDDING = "rag_allow_auto_embedding"
CONF_ALLOW_QUESTIONS = "rag_allow_questions"

CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "rag_remember_conversation_time_minutes"
CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS = "rag_remember_conversation_num_interactions"
CONF_SELECTED_LANGUAGE = "rag_selected_language"

CONF_TEMPERATURE = "rag_temperature"
CONF_K_TOP = "rag_k_top"
CONF_P_MIN = "rag_p_min"
CONF_P_TOP = "rag_p_top"
CONF_P_TYPICAL = "rag_p_typical"

TOOL_REGEX_PATTERN = re.compile(r"```homeassistant\s*(.*?)\s*```", re.DOTALL)

DEVICE_ATTRIBUTES_TO_EXCLUDE = ["friendly_name", "persistent", "supported_features"]
DEVICE_ATTRIBUTES_MAX_JSON_LENGTH = 100

DEFAULT_PROMPT = """<persona_prompt>

<instruction_prompt>

<search_fallback_prompt>

<max_retries_prompt>

<area_prompt>

{% if memory_list %}
<memories_context_prompt>
{% for memory in memory_list %}
- {{ {"memory_id": memory.id, "content": memory.content, "created_at": memory.created_at} | tojson }}
{% endfor %}
{% endif %}

<devices_prompt>
{% for device in device_list %}
- {{ {"name": device.id, "friendly_name": device.friendly_name, "aliases": device.aliases or [], "domain": device.domain or [], "device_class": device.device_class, "floor": device.floor_name, "area": device.area_name, "area_aliases": device.area_aliases or [], "floor_aliases": device.floor_aliases or [], "state": device.state, "unit_of_measurement": (device.attributes or {}).get('unit_of_measurement', device.unit_of_measurement), "attributes": device.attributes or {}} | tojson }}
{% endfor %}
"""

#-----------------------------------------------
# Default override options for new entries
#-----------------------------------------------
DEFAULT_SETTINGS = {
    CONF_ALLOW_AUTO_EMBEDDING: True,
    CONF_ALLOW_QUESTIONS: True,
    CONF_CONTEXT_LENGTH: 4096,
    CONF_EMBEDDING_BACKEND_TYPE: BACKEND_EMBEDDING_TYPE_OLLAMA,
    CONF_ENABLE_MODEL_THINKING: False,
    CONF_K_TOP: 40,
    CONF_LLM_BACKEND_TYPE: BACKEND_LLM_TYPE_OLLAMA,
    CONF_MAX_MEMORY_ENTRIES: 100,
    CONF_MAX_TOKENS: 1000,
    CONF_MAX_TOOL_CALL_ITERATIONS: 4,
    CONF_NUM_DEVICES_TO_EXTRACT: 4,
    CONF_NUM_MEMORIES_TO_EXTRACT: 4,
    CONF_NUM_TOOLS_TO_EXTRACT: 4,
    CONF_P_MIN: 0.1,
    CONF_P_TOP: 0.9,
    CONF_P_TYPICAL: 1.0,
    CONF_PROMPT: DEFAULT_PROMPT,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 30,
    CONF_RETRIEVAL_METHOD: RETRIEVAL_METHOD_AUTOMATIC,
    CONF_SELECTED_LANGUAGE: "en",
    CONF_TEMPERATURE: 0.5,
    CONF_VECTOR_DB_BACKEND_TYPE: BACKEND_VECTOR_DB_TYPE_FAISS,
    CONF_VECTOR_DB_NAME: "ha_ragent_db",
}
