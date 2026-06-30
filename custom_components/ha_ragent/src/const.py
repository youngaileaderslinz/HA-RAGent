import re

#-----------------------------------------------
# General constants
#-----------------------------------------------
DOMAIN = "ha_ragent"
RAGENT_LLM_API_ID = "ha_ragent_api"
RAGENT_LLM_API_NAME = "HA RAGent"
RAGENT_SEARCH_DEVICES_TOOL_NAME = "HassSearchDevices"
RAGENT_SEARCH_TOOLS_TOOL_NAME = "HassSearchTools"
INTEGRATION_VERSION = "0.3.0"

STARTUP_EMBEDDING_RUNNING_FLAG = "ha_ragent_startup_embedding_running"

#-----------------------------------------------
# Language constants
#-----------------------------------------------
CONF_SELECTED_LANGUAGE = "selected_language"

SELECTED_LANGUAGE_OPTIONS = [ "en", "de" ]

DEFAULT_LANGUAGE = "en"

#-----------------------------------------------
# Service Tool constants
#-----------------------------------------------
RAGENT_TIMER_DEVICE_ID = "ha_ragent_timer_device_a03a100a-81ca-415d"

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

DEFAULT_VECTOR_DB_BACKEND_TYPE = BACKEND_VECTOR_DB_TYPE_FAISS
DEFAULT_VECTOR_DB_NAME = "ha_ragent_db"

#-----------------------------------------------
# Embedding backend constants
#-----------------------------------------------
CONF_EMBEDDING_BACKEND_TYPE = "rag_embedding_backend"
CONF_EMBEDDING_MODEL = "rag_embedding_model"
CONF_EMBEDDING_HOST = "rag_embedding_host"
CONF_EMBEDDING_PORT = "rag_embedding_port"
CONF_EMBEDDING_SSL = "rag_embedding_ssl"

BACKEND_EMBEDDING_TYPE_OLLAMA = "ollama"

BACKEND_EMBEDDING_TYPE_OPTIONS = [ 
    BACKEND_EMBEDDING_TYPE_OLLAMA 
]

DEFAULT_EMBEDDING_BACKEND_TYPE = BACKEND_EMBEDDING_TYPE_OLLAMA

#-----------------------------------------------
# Chat backend constants
#-----------------------------------------------
CONF_LLM_BACKEND_TYPE = "rag_llm_backend"
CONF_LLM_MODEL = "rag_llm_model"
CONF_LLM_HOST = "rag_llm_host"
CONF_LLM_PORT = "rag_llm_port"
CONF_LLM_SSL = "rag_llm_ssl"

BACKEND_LLM_TYPE_OLLAMA = "ollama"

BACKEND_LLM_TYPE_OPTIONS = [ 
    BACKEND_LLM_TYPE_OLLAMA 
]

DEFAULT_LLM_BACKEND_TYPE = BACKEND_LLM_TYPE_OLLAMA

#-----------------------------------------------
# Prompt configuration constants
#----------------------------------------------
CONF_NUM_DEVICES_TO_EXTRACT = "rag_num_devices_to_extract"
CONF_NUM_TOOLS_TO_EXTRACT = "rag_num_tools_to_extract"
CONF_CONTEXT_LENGTH = "rag_context_length"

CONF_MAX_TOKENS = "rag_max_tokens"
CONF_MAX_TOOL_CALL_ITERATIONS = "rag_max_tool_call_iterations"

CONF_PROMPT = "rag_prompt"

CONF_ENABLE_MODEL_THINKING = "rag_enable_model_thinking"

CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "rag_remember_conversation_time_minutes"
CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS = "rag_remember_conversation_num_interactions"
CONF_SELECTED_LANGUAGE = "rag_selected_language"

CONF_TEMPERATURE = "rag_temperature"
CONF_K_TOP = "rag_k_top"
CONF_P_MIN = "rag_p_min"
CONF_P_TOP = "rag_p_top"
CONF_P_TYPICAL = "rag_p_typical"

PERSONA_PROMPTS = {
    "de": "Du bist \"YAIL\", ein hilfreicher KI-Assistent, der die Geräte in einem Haus steuert. Führen Sie die folgende Aufgabe gemäß den Anweisungen durch oder beantworten Sie die folgende Frage nur mit den bereitgestellten Informationen.",
    "en": "You are 'YAIL', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.",
}
CURRENT_DATE_PROMPT = {
    "de": """{% set day_name = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"] %}{% set month_name = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"] %}Die aktuelle Uhrzeit und das aktuelle Datum sind {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}.""",
    "en": """The current time and date is {{ (as_timestamp(now()) | timestamp_custom("%I:%M %p on %A %B %d, %Y", True, "")) }}"""
}
DEVICES_PROMPT = {
    "de": "## Verfügbare Geräte:",
    "en": "## Available Devices:",
}
AREAS_PROMPT = {
    "de": """Bereichsanweisungen:
{% if area_name %}
- Aktueller Standort: Du befindest dich physisch im {{ area_name }}{% if floor_name %} ({{ floor_name }} Stock){% endif %}.
- Standardverhalten: Wenn der Benutzer eine Gerätekategorie angibt (z. B. „die Lichter“), ohne einen Raum zu nennen, ziele NUR auf die Geräte im {{ area_name }} ab.
{% else %}
- KRITISCH: Du hast keine Erlaubnis, einen Raum zu erraten oder das gesamte Haus anzusprechen.
- Wenn der Benutzer keinen Raum angibt, MUSST du um Klarstellung bitten.
{% endif %}""",
    "en": """Area Instructions:
{% if area_name %}
- Current Location: You are physically located in the {{ area_name }}{% if floor_name %} ({{ floor_name }} floor){% endif %}.
- Default Behavior: If the user specifies a device category (e.g., "the lights") without naming a room, target ONLY the devices within the {{ area_name }}.
{% else %}
- CRITICAL: You do not have permission to guess a room or target the entire house.
- If the user does not name a room, you MUST ask for clarification.
{% endif %}"""
}

USER_INSTRUCTION = {
    "de": "## Benutzeranweisung:",
    "en": "## User instruction:"
}


DEVICE_CONTROL_PROMPT = {
    "de": """## Geräte Steuerungsanweisungen:
1. Auflösung
- Nutze zuerst die neueste Benutzernachricht. Älterer Verlauf ist nur Hilfskontext.
- Wenn die neueste Nachricht eine Folgeanweisung wie "auch", "dann", "zusätzlich" oder eine weitere direkte Aktion enthält, behandle sie als neue auszufuehrende Steuerungsanweisung.
- Löse Ziele ueber Name, entity_id, Domain, device_class und Bereich auf.
- Wenn Bereich und Kategorie genannt sind, nimm alle passenden Geräte in diesem Bereich.
- Steuere nie irrelevante Geräte oder Geräte aus nicht genannten Bereichen.
- Nutze `HassSearchDevices` nur als Fallback zur Auflösung, nicht um erst Optionen vorzuschlagen oder um Erlaubnis für eine bereits klare Steuerungsanweisung zu erfragen. Verwende Treffer danach wie normale Geräte und bevorzuge ihre exakte `entity_id`.
2. Tool-Aufrufe
- Bei mehreren Treffern gib pro Gerät einen eigenen `homeassistant`-Block aus.
- Gib erst alle Tool-Blöcke aus, danach kurzen natürlichen Text.
3. Antworten
- Behaupte nie Erfolg ohne Tool-Aufruf und bestätige Erfolg nur aus echten Tool-Ergebnissen.
- Bei Folgeanweisungen beziehe dich im Text nur auf die neueste Aktion und wiederhole keine früheren Bereiche oder Geräte, ausser der Benutzer verlangt eine Gesamtsummary.
- Wenn nichts passt oder das Ziel unklar ist, antworte kurz oder frage nach Klarstellung.
- Nutze im Text nur freundliche Namen, keine technischen IDs.""",
    "en": """## Device Control Instructions:
1. Resolution
- Use the latest user message first. Older conversation is supporting context only.
- If the latest message is a follow-up command like "also", "then", or another direct action, treat it as a new command to execute.
- Resolve targets by name, entity_id, domain, device_class, and area.
- If the user names an area and a category, include all matching devices in that area.
- If you only see one matching device but the request sounds like a category or room-wide action, you may use `HassSearchDevices` once to check whether more matching devices exist.
- Never control irrelevant devices or devices from areas the user did not mention.
- Use `HassSearchDevices` only as a fallback for resolution, not to preview options or ask permission for an already clear control request. Treat returned devices like normal available devices and prefer their exact `entity_id`.
2. Tool Calls
- If multiple devices match, emit one `homeassistant` block per device.
- Output all tool blocks first, then the short natural-language response.
3. Responses
- Never claim an action happened without a tool call, and confirm success only from real tool results.
- For follow-up commands, talk only about the newest action and do not repeat earlier rooms or devices unless the user asks for a full summary.
- If nothing matches or the target is unclear, reply briefly or ask for clarification.
- Use friendly names in text, never technical IDs."""
}

CONVERSATION_PRIORITY_PROMPT = {
    "de": """Die neueste Benutzernachricht hat Priorität. Direkte Folgeanweisungen sind auszuführende Befehle, keine Bitte um Bestätigung. Antworte bei Folgeanweisungen nur über die neueste Aktion. Nutze `HassSearchDevices` nur als Fallback zur Auflösung. Simuliere keine erfolgreiche Gerätesteuerung: gib Tool-Aufrufe aus, frage nur bei echter Unklarheit nach oder antworte auf Basis echter Tool-Ergebnisse.""",
    "en": """The latest user message has priority. Direct follow-up commands should be executed, not turned into confirmation questions. For follow-up commands, respond only about the newest action. Use `HassSearchDevices` only as a fallback for resolution. Do not simulate successful device control: emit tool calls, ask only when genuinely unclear, or respond from real tool results.""",
}

DEVICE_ATTRIBUTES_TO_EXCLUDE = ["friendly_name", "persistent", "supported_features"]
DEVICE_ATTRIBUTES_MAX_JSON_LENGTH = 100

TOOL_REGEX_PATTERN = re.compile(r"```homeassistant\s*(.*?)\s*```", re.DOTALL)

DEFAULT_NUM_DEVICES_TO_EXTRACT = 4
DEFAULT_NUM_TOOLS_TO_EXTRACT = 4
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 8

DEFAULT_PROMPT = """<persona>
<current_date>
<area_prompt>

<device_control_prompt>

<conversation_priority_prompt>

<devices>
{% for device in device_list %}
- { "name": "{{ device.id }}", "friendly_name": "{{ device.name }}", "aliases": {{ device.aliases | tojson }}, "domain": {{ device.domain | tojson }}, "area": "{{ device.area_name }}", "device_class": {{ device.domain | tojson }}, "state": {{ device.state }} }
{% endfor %}

<user_instruction>
"""

DEFAULT_ENABLE_MODEL_THINKING = False
DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES = 5
DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS = 10
DEFAULT_SELECTED_LANGUAGE = "en"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_K_TOP = 40
DEFAULT_P_MIN = 0.1
DEFAULT_P_TOP = 0.9
DEFAULT_P_TYPICAL = 1.0

#-----------------------------------------------
# Default override options for new entries
#-----------------------------------------------
DEFAULT_OPTIONS = {
    CONF_PROMPT: DEFAULT_PROMPT,
    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    CONF_K_TOP: DEFAULT_K_TOP,
    CONF_P_TOP: DEFAULT_P_TOP,
    CONF_P_MIN: DEFAULT_P_MIN,
    CONF_P_TYPICAL: DEFAULT_P_TYPICAL,
    CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES: DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
    CONF_NUM_DEVICES_TO_EXTRACT: DEFAULT_NUM_DEVICES_TO_EXTRACT,
}
